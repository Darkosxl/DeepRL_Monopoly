"""
Fixed-policy baseline agents — 4 distinct personality archetypes.

Each agent has a well-defined behavioural identity expressed across every
decision point: buying, building, trading, mortgaging, and responding to
incoming offers.  The personalities are deliberately contrasting so the
learning agent faces meaningfully different opponents.

─────────────────────────────────────────────────────────────────────────────
  Agent                  │ Core trait
─────────────────────────┼───────────────────────────────────────────────────
  TheHoarder      (FP-A) │ Cash-first. Buys only sure things, never trades.
  TheDealMaker    (FP-B) │ Trade-obsessed. Spams offers, rejects all incoming.
  TheGambler      (FP-C) │ Buys everything recklessly, skips building.
  TheBuilder      (FP-D) │ Selective buyer, goes deep on development ASAP.
  TheBlocker      (FP-E) | It checks whether purchasing would prevent any opponent from completing a colour group
  The RailBaron   (FP-E) | It sends buy-offers for any railroad or utility held by an opponent at market price
─────────────────────────────────────────────────────────────────────────────
"""

import random
from typing import List, Optional

from .actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    TRADE_CASH_LEVELS,
    ActionType,
    AuctionAction,
)
from .constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    NUM_PLAYERS,
    PROPERTIES,
    PROPERTY_IDS,
    RAILROAD_IDS,
    REAL_ESTATE_IDS,
    UTILITY_IDS,
)
from .env import PHASE_AUCTION, MonopolyEnv, TradeOffer

# ── Shared helpers ────────────────────────────────────────────────────────────


def _incoming_offer(env: MonopolyEnv, pid: int) -> Optional[TradeOffer]:
    """Return the first pending trade whose recipient is pid, or None."""
    return env._incoming_trade(pid)


def _buy_trade_action(
    pid: int,
    target_pid: int,
    sq: int,
    price_idx: int,
    env: MonopolyEnv,
    allowed: List[int],
) -> Optional[int]:
    """
    Return the buy-trade action index for pid offering cash to target_pid
    for property at square sq, at TRADE_CASH_LEVELS[price_idx].
    Returns None if the action is not in allowed.
    """
    others = [i for i in range(NUM_PLAYERS) if i != pid]
    if target_pid not in others:
        return None
    t_idx = others.index(target_pid)
    prop_idx = PROPERTY_IDS.index(sq)
    action = (
        OFFSETS["buy_trade"]
        + t_idx * len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)
        + prop_idx * len(TRADE_CASH_LEVELS)
        + price_idx
    )
    return action if action in allowed else None


def _sell_trade_action(
    pid: int,
    target_pid: int,
    sq: int,
    price_idx: int,
    env: MonopolyEnv,
    allowed: List[int],
) -> Optional[int]:
    """Return the sell-trade action index (we offer our property for cash)."""
    others = [i for i in range(NUM_PLAYERS) if i != pid]
    if target_pid not in others:
        return None
    t_idx = others.index(target_pid)
    prop_idx = PROPERTY_IDS.index(sq)
    action = (
        OFFSETS["sell_trade"]
        + t_idx * len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)
        + prop_idx * len(TRADE_CASH_LEVELS)
        + price_idx
    )
    return action if action in allowed else None


def _exchange_action(
    pid: int,
    target_pid: int,
    offer_sq: int,
    req_sq: int,
    env: MonopolyEnv,
    allowed: List[int],
) -> Optional[int]:
    """Return the property-swap action index."""
    others = [i for i in range(NUM_PLAYERS) if i != pid]
    if target_pid not in others:
        return None
    t_idx = others.index(target_pid)
    n = len(PROPERTY_IDS)
    offer_idx = PROPERTY_IDS.index(offer_sq)
    req_raw = PROPERTY_IDS.index(req_sq)
    req_enc = req_raw if req_raw < offer_idx else req_raw - 1
    action = OFFSETS["exch_trade"] + t_idx * n * (n - 1) + offer_idx * (n - 1) + req_enc
    return action if action in allowed else None


# ── Base class ────────────────────────────────────────────────────────────────


class FixedPolicyAgent:
    """
    Minimal base class. Concrete agents override every decision method.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id

    def choose_action(self, env: MonopolyEnv) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        player = env.players[self.player_id]

        if env.phase == PHASE_AUCTION:
            return self._auction_action(allowed, env)

        # 1. Respond to incoming trade first (highest priority)
        offer = _incoming_offer(env, self.player_id)
        if offer is not None:
            return (
                int(ActionType.ACCEPT_TRADE)
                if self._should_accept_trade(offer, env)
                else int(ActionType.DECLINE_TRADE)
            )

        # 2. Jail escape
        if player.in_jail:
            action = self._handle_jail(allowed, player)
            if action is not None:
                return action

        # 3. Buy property on landing
        if int(ActionType.BUY_PROPERTY) in allowed:
            sq = player.position
            prop = env.properties.get(sq)
            if prop and prop.owner is None and self._should_buy(player, prop, env):
                return int(ActionType.BUY_PROPERTY)

        # 4. Build houses / hotels
        build = self._best_build_action(allowed, env)
        if build is not None:
            return build

        # 5. Initiate trades
        trade = self._make_trade_offer(allowed, env)
        if trade is not None:
            return trade

        # 6. Mortgage if necessary
        mort = self._maybe_mortgage(allowed, env)
        if mort is not None:
            return mort

        # 7. Roll dice
        if int(ActionType.ROLL_DICE) in allowed:
            return int(ActionType.ROLL_DICE)

        return int(ActionType.END_TURN)

    def _auction_action(self, allowed: List[int], env: MonopolyEnv) -> int:
        """Bid up to list price when this personality would buy the deed."""
        prop = env.properties.get(env.auction_property_id)
        player = env.players[self.player_id]
        if prop is None or not self._should_buy(player, prop, env):
            return int(AuctionAction.PASS)

        bids = [
            int(action)
            for action, increment in AUCTION_ACTION_TO_INCREMENT.items()
            if int(action) in allowed and env.auction_high_bid + increment <= prop.price
        ]
        if not bids:
            return int(AuctionAction.PASS)
        return max(
            bids,
            key=lambda action: AUCTION_ACTION_TO_INCREMENT[AuctionAction(action)],
        )

    def _should_accept_trade(self, offer: TradeOffer, env: MonopolyEnv) -> bool:
        raise NotImplementedError

    def _handle_jail(self, allowed: List[int], player) -> Optional[int]:
        raise NotImplementedError

    def _should_buy(self, player, prop, env: MonopolyEnv) -> bool:
        raise NotImplementedError

    def _best_build_action(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        raise NotImplementedError

    def _make_trade_offer(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        raise NotImplementedError

    def _maybe_mortgage(self, allowed: List[int], env: MonopolyEnv) -> Optional[int]:
        raise NotImplementedError


# =============================================================================
#  Agent 1 — TheHoarder
# =============================================================================


class TheHoarder(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : only if completing a monopoly OR it's a railroad AND cash
                 stays above a $600 safety buffer.
    Building   : never — prefers cash in hand.
    Trading    : never initiates. Refuses every incoming offer.
    Jail       : pays bail immediately (stay liquid, don't waste turns).
    Mortgaging : very quick to mortgage anything that isn't a monopoly or
                 railroad once cash dips below $400.
    """

    _CASH_FLOOR = 400
    _BUY_BUFFER = 600

    def _should_accept_trade(self, offer, env):
        return False

    def _handle_jail(self, allowed, player):
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        if int(ActionType.PAY_BAIL) in allowed:
            return int(ActionType.PAY_BAIL)
        return None

    def _should_buy(self, player, prop, env):
        if not player.can_afford(prop.price + self._BUY_BUFFER):
            return False
        if prop.color == "railroad":
            return True
        color = prop.color
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned = sum(1 for s in group if env.properties[s].owner == self.player_id)
            if owned + 1 == len(group):
                return True
        return False

    def _best_build_action(self, allowed, env):
        return None

    def _make_trade_offer(self, allowed, env):
        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        if player.cash >= self._CASH_FLOOR:
            return None
        mortgage_order = (
            list(UTILITY_IDS)
            + [
                p
                for p in PROPERTY_IDS
                if PROPERTIES[p]["color"] not in ("railroad", "utility")
            ]
            + list(RAILROAD_IDS)
        )
        for sq in mortgage_order:
            prop = env.properties.get(sq)
            if prop is None or prop.owner != self.player_id or prop.is_monopoly:
                continue
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None


# =============================================================================
#  Agent 2 — TheDealMaker
# =============================================================================


class TheDealMaker(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : standard — buys if affordable with a small $100 buffer.
    Building   : only after accumulating $800+ above the house price.
    Trading    : initiates aggressively — bargain buy-offers, exchanges,
                 and sells own non-monopoly props at a premium.
    Incoming   : always declines — never accepts other people's terms.
    Jail       : uses GOOJ card if held; otherwise waits (saves cash).
    Mortgaging : mortgages non-monopoly, non-railroad props when cash < $300.
    """

    _BUY_BUFFER = 100
    _BUILD_BUFFER = 800

    def _should_accept_trade(self, offer, env):
        return False

    def _handle_jail(self, allowed, player):
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        return None

    def _should_buy(self, player, prop, env):
        return player.can_afford(prop.price + self._BUY_BUFFER)

    def _best_build_action(self, allowed, env):
        player = env.players[self.player_id]
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            hp = prop.data["house_price"]
            if not player.can_afford(hp + self._BUILD_BUFFER):
                continue
            for action_key in ("improve_house", "improve_hotel"):
                action = OFFSETS[action_key] + i
                if action in allowed:
                    return action
        return None

    def _make_trade_offer(self, allowed, env):
        pid = self.player_id
        others = [
            i for i in range(NUM_PLAYERS) if i != pid and not env.players[i].bankrupt
        ]

        # 1. Bargain buy-offer for a colour piece one step from monopoly (0.75x)
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned = [s for s in group if env.properties[s].owner == pid]
            need = [
                s
                for s in group
                if env.properties[s].owner not in (pid, None)
                and not env.players[env.properties[s].owner].bankrupt
            ]
            if len(owned) + 1 == len(group) and need:
                sq = need[0]
                target = env.properties[sq].owner
                action = _buy_trade_action(pid, target, sq, 0, env, allowed)
                if action is not None:
                    return action

        # 2. Exchange own non-monopoly prop for a needed colour piece
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned_here = [s for s in group if env.properties[s].owner == pid]
            if not owned_here:
                continue
            need = [
                s
                for s in group
                if env.properties[s].owner not in (pid, None)
                and not env.players[env.properties[s].owner].bankrupt
            ]
            if not need:
                continue
            for req_sq in need:
                target = env.properties[req_sq].owner
                for offer_sq in owned_here:
                    if not env.properties[offer_sq].is_monopoly:
                        action = _exchange_action(
                            pid, target, offer_sq, req_sq, env, allowed
                        )
                        if action is not None:
                            return action

        # 3. Sell any non-monopoly property at a premium (1.25x)
        for sq in PROPERTY_IDS:
            prop = env.properties[sq]
            if prop.owner != pid or prop.is_monopoly or prop.houses > 0:
                continue
            for target in others:
                action = _sell_trade_action(pid, target, sq, 2, env, allowed)
                if action is not None:
                    return action

        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        if player.cash >= 300:
            return None
        for sq in PROPERTY_IDS:
            prop = env.properties.get(sq)
            if (
                prop is None
                or prop.owner != self.player_id
                or prop.is_monopoly
                or PROPERTIES[sq]["color"] == "railroad"
            ):
                continue
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None


# =============================================================================
#  Agent 3 — TheGambler
# =============================================================================


class TheGambler(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : buys every unowned property as long as $50 remains in cash.
    Building   : only when cash is extremely comfortable ($1000+ surplus).
    Trading    : makes quick monopoly-completing buy-offers at market price;
                 accepts trades that give a monopoly or are near break-even.
    Jail       : always pays bail — hates missing landing opportunities.
    Mortgaging : only in last-gasp situations (cash < $100), mortgages
                 cheapest asset first.
    """

    _COMFORTABLE_CASH = 1000
    _DESPERATION_CASH = 100

    def _should_accept_trade(self, offer, env):
        pid = self.player_id
        if offer.offered_prop:
            color = offer.offered_prop.color
            group = COLOR_GROUPS.get(color, [])
            if group:
                would_own = sum(1 for s in group if env.properties[s].owner == pid) + 1
                if would_own == len(group):
                    return True
        # Accept if roughly break-even (tolerate up to $50 loss)
        return offer.net_worth() >= -50

    def _handle_jail(self, allowed, player):
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        if int(ActionType.PAY_BAIL) in allowed:
            return int(ActionType.PAY_BAIL)
        return None

    def _should_buy(self, player, prop, env):
        return player.can_afford(prop.price + 50)

    def _best_build_action(self, allowed, env):
        player = env.players[self.player_id]
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            hp = prop.data["house_price"]
            if not player.can_afford(hp + self._COMFORTABLE_CASH):
                continue
            for action_key in ("improve_house", "improve_hotel"):
                action = OFFSETS[action_key] + i
                if action in allowed:
                    return action
        return None

    def _make_trade_offer(self, allowed, env):
        pid = self.player_id
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            owned = [s for s in group if env.properties[s].owner == pid]
            need = [
                s
                for s in group
                if env.properties[s].owner not in (pid, None)
                and not env.players[env.properties[s].owner].bankrupt
            ]
            if len(owned) + 1 == len(group) and len(need) == 1:
                sq = need[0]
                target = env.properties[sq].owner
                action = _buy_trade_action(pid, target, sq, 1, env, allowed)
                if action is not None:
                    return action
        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        if player.cash >= self._DESPERATION_CASH:
            return None
        candidates = [
            sq
            for sq in PROPERTY_IDS
            if (
                env.properties.get(sq) is not None
                and env.properties[sq].owner == self.player_id
                and not env.properties[sq].is_monopoly
                and env.properties[sq].houses == 0
            )
        ]
        candidates.sort(key=lambda sq: PROPERTIES[sq]["mortgage"])
        for sq in candidates:
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None


# =============================================================================
#  Agent 4 — TheBuilder
# =============================================================================


class TheBuilder(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : only buys green + dark-blue real estate and railroads.
                 Passes on everything else to save capital for development.
    Building   : highest priority. Mortgages non-target assets to fund houses.
    Trading    : buy-offers for target-colour pieces at a premium (1.25x).
                 Accepts trades that hand it a target-colour monopoly.
    Jail       : waits the full 3 turns (free rest; opponents still pay rent).
    Mortgaging : mortgages non-target, non-monopoly props freely;
                 cash floor of only $50 when building.
    """

    _TARGET_COLORS = {"green", "darkblue"}
    _BUILD_CASH_FLOOR = 50
    _MORTGAGE_TRIGGER = 200

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self._target_squares = [
            sq for sq in PROPERTY_IDS if PROPERTIES[sq]["color"] in self._TARGET_COLORS
        ]

    def _should_accept_trade(self, offer, env):
        pid = self.player_id
        if offer.offered_prop:
            color = offer.offered_prop.color
            if color in self._TARGET_COLORS:
                group = COLOR_GROUPS[color]
                would_own = sum(1 for s in group if env.properties[s].owner == pid) + 1
                if would_own == len(group):
                    return True
        return False

    def _handle_jail(self, allowed, player):
        # Only use GOOJ card; never pay bail
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        return None

    def _should_buy(self, player, prop, env):
        if not player.can_afford(prop.price + 200):
            return False
        if prop.color == "railroad":
            return True
        return prop.color in self._TARGET_COLORS

    def _best_build_action(self, allowed, env):
        player = env.players[self.player_id]
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = env.properties[sq]
            if prop.owner != self.player_id or not prop.is_monopoly:
                continue
            if prop.color not in self._TARGET_COLORS:
                continue
            hp = prop.data["house_price"]
            if player.can_afford(hp + self._BUILD_CASH_FLOOR):
                for action_key in ("improve_hotel", "improve_house"):
                    action = OFFSETS[action_key] + i
                    if action in allowed:
                        return action
            else:
                mort = self._mortgage_for_build(allowed, env)
                if mort is not None:
                    return mort
        return None

    def _mortgage_for_build(self, allowed, env) -> Optional[int]:
        for sq in PROPERTY_IDS:
            prop = env.properties.get(sq)
            if (
                prop is None
                or prop.owner != self.player_id
                or prop.is_monopoly
                or PROPERTIES[sq]["color"] in self._TARGET_COLORS
                or prop.houses > 0
            ):
                continue
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None

    def _make_trade_offer(self, allowed, env):
        pid = self.player_id
        for color in self._TARGET_COLORS:
            group = COLOR_GROUPS[color]
            owned = [s for s in group if env.properties[s].owner == pid]
            need = [
                s
                for s in group
                if env.properties[s].owner not in (pid, None)
                and not env.players[env.properties[s].owner].bankrupt
            ]
            if len(owned) + 1 == len(group) and need:
                sq = need[0]
                target = env.properties[sq].owner
                action = _buy_trade_action(pid, target, sq, 2, env, allowed)
                if action is not None:
                    return action
        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        if player.cash >= self._MORTGAGE_TRIGGER:
            return None
        return self._mortgage_for_build(allowed, env)


# =============================================================================
#  Agent 5 — TheBlocker
#  Personality: threat-aware saboteur. Watches which colour groups opponents
#  are close to monopolising and specifically buys or holds those missing
#  pieces to deny them. Willing to sit on "useless" properties for the whole
#  game just to prevent an opponent from completing a set. Accepts trades
#  only when they let it acquire a blocking piece. Never builds.
# =============================================================================


class TheBlocker(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : always buys a property if doing so denies an opponent a
                 monopoly (blocking priority). Falls back to normal buying
                 with a $300 buffer when no blocking opportunity exists.
    Building   : never — capital is held in reserve for opportunistic blocking
                 purchases and bail payments.
    Trading    : never initiates trades (giving a property away defeats the
                 whole point). Accepts incoming offers ONLY when the property
                 it receives is itself a blocking piece against another player.
    Jail       : pays bail immediately — must stay mobile to land on and
                 snap up blocking pieces.
    Mortgaging : mortgages own non-blocking, non-monopoly properties when
                 cash drops below $350.
    """

    _BUY_BUFFER = 300
    _MORTGAGE_FLOOR = 350

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _blocking_squares(self, env: MonopolyEnv) -> set:
        """
        Return the set of unowned squares whose purchase would prevent any
        opponent from completing a colour-group monopoly.

        A square is a blocking target when:
          - It is currently unowned (bank), AND
          - At least one opponent already owns at least one other property
            in the same colour group.
        """
        pid = self.player_id
        targets = set()
        for color, group in COLOR_GROUPS.items():
            if color in ("railroad", "utility"):
                continue
            unowned = [s for s in group if env.properties[s].owner is None]
            if not unowned:
                continue
            # Is any opponent invested in this group?
            opponent_in_group = any(
                env.properties[s].owner not in (None, pid) for s in group
            )
            if opponent_in_group:
                targets.update(unowned)
        return targets

    def _is_blocking_prop(self, sq: int, env: MonopolyEnv) -> bool:
        return sq in self._blocking_squares(env)

    # ── Decision methods ──────────────────────────────────────────────────────

    def _should_accept_trade(self, offer, env):
        # Accept only if the property we'd receive is a blocking piece
        if offer.offered_prop is None:
            return False
        return self._is_blocking_prop(offer.offered_prop.square_id, env)

    def _handle_jail(self, allowed, player):
        # Must stay mobile — pay bail to keep landing on properties
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        if int(ActionType.PAY_BAIL) in allowed:
            return int(ActionType.PAY_BAIL)
        return None

    def _should_buy(self, player, prop, env):
        if not player.can_afford(prop.price + self._BUY_BUFFER):
            return False
        # Top priority: this purchase blocks an opponent's monopoly
        if self._is_blocking_prop(prop.square_id, env):
            # Relax buffer for blocking buys — worth stretching for
            return player.can_afford(prop.price + 50)
        # Secondary: normal purchase with standard buffer
        return player.cash >= prop.price + self._BUY_BUFFER

    def _best_build_action(self, allowed, env):
        # Never builds
        return None

    def _make_trade_offer(self, allowed, env):
        # Never gives up any property
        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        if player.cash >= self._MORTGAGE_FLOOR:
            return None
        pid = self.player_id
        blocking = self._blocking_squares(env)

        # Mortgage in order: non-blocking, non-monopoly properties first,
        # preserving any blocking pieces and monopolies to the last.
        candidates = []
        for sq in PROPERTY_IDS:
            prop = env.properties.get(sq)
            if (
                prop is None
                or prop.owner != pid
                or prop.is_monopoly
                or prop.houses > 0
                or sq in blocking
            ):
                continue
            candidates.append(sq)

        # If nothing safe to mortgage, reluctantly consider blocking pieces
        if not candidates:
            candidates = [
                sq
                for sq in PROPERTY_IDS
                if (
                    env.properties.get(sq) is not None
                    and env.properties[sq].owner == pid
                    and not env.properties[sq].is_monopoly
                    and env.properties[sq].houses == 0
                )
            ]

        for sq in candidates:
            idx = PROPERTY_IDS.index(sq)
            action = OFFSETS["mortgage"] + idx
            if action in allowed:
                return action
        return None


# =============================================================================
#  Agent 6 — TheRailBaron
#  Personality: pure passive-income machine. Targets all 4 railroads and
#  both utilities exclusively. Never touches colour-group real estate at all.
#  Once it owns the full railroad set (rent $200/visit) and both utilities
#  (10× dice roll) it just rolls dice and watches rent accumulate. Trades
#  aggressively to consolidate the infrastructure set but has no interest
#  in colour properties whatsoever.
# =============================================================================


class TheRailBaron(FixedPolicyAgent):
    """
    Traits
    ------
    Buying     : ONLY railroads and utilities. Hard pass on all colour-group
                 real estate regardless of price or circumstance.
    Building   : never — railroads and utilities cannot be developed.
    Trading    : actively tries to acquire missing railroads via buy-offers
                 at market price (1.0×). Will sell its own utilities cheaply
                 (0.75×) if it already owns both, to get cash for railroads.
                 Accepts incoming trades only if they hand it a railroad or
                 utility it doesn't yet own.
    Jail       : uses GOOJ card if available; otherwise waits (rent still
                 comes in while sitting in jail, so it's not urgent).
    Mortgaging : mortgages utilities (low priority once all railroads owned)
                 then sits tight. Never mortgages railroads.
    """

    _INFRA_SQUARES = set(RAILROAD_IDS) | set(UTILITY_IDS)
    _BUY_BUFFER = 150

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _missing_infra(self, env: MonopolyEnv) -> list:
        """Return infrastructure squares not yet owned by this agent."""
        pid = self.player_id
        return [sq for sq in self._INFRA_SQUARES if env.properties[sq].owner != pid]

    def _opponent_owns(self, sq: int, env: MonopolyEnv) -> Optional[int]:
        """Return the owner pid of sq if it's an opponent, else None."""
        pid = self.player_id
        owner = env.properties[sq].owner
        return owner if (owner is not None and owner != pid) else None

    # ── Decision methods ──────────────────────────────────────────────────────

    def _should_accept_trade(self, offer, env):
        # Accept only if we receive a railroad or utility we don't yet own
        if offer.offered_prop is None:
            return False
        sq = offer.offered_prop.square_id
        if sq not in self._INFRA_SQUARES:
            return False
        return env.properties[sq].owner != self.player_id

    def _handle_jail(self, allowed, player):
        # Use GOOJ card if available; otherwise wait (rent flows regardless)
        if int(ActionType.USE_GOOJ_CARD) in allowed:
            return int(ActionType.USE_GOOJ_CARD)
        return None

    def _should_buy(self, player, prop, env):
        # Only infrastructure; skip everything else completely
        if prop.square_id not in self._INFRA_SQUARES:
            return False
        return player.can_afford(prop.price + self._BUY_BUFFER)

    def _best_build_action(self, allowed, env):
        # Railroads and utilities cannot be developed
        return None

    def _make_trade_offer(self, allowed, env):
        pid = self.player_id
        player = env.players[pid]

        # 1. Buy-offer for any railroad we're missing, at market price (1.0×)
        for sq in RAILROAD_IDS:
            target = self._opponent_owns(sq, env)
            if target is None:
                continue
            if env.players[target].bankrupt:
                continue
            action = _buy_trade_action(pid, target, sq, 1, env, allowed)
            if action is not None:
                return action

        # 2. Buy-offer for missing utilities at market price
        for sq in UTILITY_IDS:
            target = self._opponent_owns(sq, env)
            if target is None:
                continue
            if env.players[target].bankrupt:
                continue
            action = _buy_trade_action(pid, target, sq, 1, env, allowed)
            if action is not None:
                return action

        # 3. If we already own both utilities, sell one cheaply (0.75×)
        #    to raise cash for a railroad buy-offer
        owned_utils = [sq for sq in UTILITY_IDS if env.properties[sq].owner == pid]
        missing_rails = [sq for sq in RAILROAD_IDS if env.properties[sq].owner != pid]
        if len(owned_utils) == 2 and missing_rails:
            sell_sq = owned_utils[0]
            others = [
                i
                for i in range(NUM_PLAYERS)
                if i != pid and not env.players[i].bankrupt
            ]
            for target in others:
                action = _sell_trade_action(pid, target, sell_sq, 0, env, allowed)
                if action is not None:
                    return action

        return None

    def _maybe_mortgage(self, allowed, env):
        player = env.players[self.player_id]
        pid = self.player_id
        # Only mortgage utilities (never railroads — they are the whole strategy)
        for sq in UTILITY_IDS:
            prop = env.properties.get(sq)
            if prop is None or prop.owner != pid:
                continue
            # Only mortgage a utility if we're cash-poor AND own both
            # (one utility alone is still valuable; owning both unlocks 10× rent)
            owned_utils = sum(1 for s in UTILITY_IDS if env.properties[s].owner == pid)
            if player.cash < 200 and owned_utils == 2:
                idx = PROPERTY_IDS.index(sq)
                action = OFFSETS["mortgage"] + idx
                if action in allowed:
                    return action
        return None


# ── Registry ──────────────────────────────────────────────────────────────────

FP_AGENT_CLASSES = [
    TheHoarder,
    TheDealMaker,
    TheGambler,
    TheBuilder,
    TheBlocker,
    TheRailBaron,
]

# Backward-compatible aliases
FPAgentA = TheHoarder
FPAgentB = TheDealMaker
FPAgentC = TheGambler
FPAgentD = TheBuilder
FPAgentE = TheBlocker
FPAgentF = TheRailBaron
