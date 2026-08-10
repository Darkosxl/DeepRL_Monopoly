"""
Monopoly Simulator - FIXED VERSION
===================================
Key fixes:
  - Phase state machine now correctly sequences pre-roll → roll → post-roll → next player
  - get_allowed_actions() properly gates actions by phase and whose turn it is
  - Bankruptcy correctly eliminates players
  - Turn advancement works cleanly across all phases
"""

""" Fix Trade Order """

import random
from typing import Dict, List, Optional

import numpy as np

from .actions import (
    ACTION_SPACE_SIZE,
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    PROPERTY_IDS,
    ActionType,
    AuctionAction,
)
from .constants import (
    BOARD,
    COLOR_GROUPS,
    FREE_PARKING,
    GO_SALARY,
    GO_TO_JAIL_SQUARE,
    HOTEL_SUPPLY,
    HOUSE_SUPPLY,
    INCOME_TAX_SQUARE,
    JAIL_BAIL,
    JAIL_SQUARE,
    LUXURY_TAX_SQUARE,
    MAX_HOUSES,
    MAX_JAIL_TURNS,
    NUM_PLAYERS,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    STARTING_CASH,
    TRADE_CASH_LEVELS,
)
from .state import Player, Property, build_state_vector


class TradeOffer:
    def __init__(
        self,
        from_player,
        to_player,
        offered_prop=None,
        requested_prop=None,
        cash_offered=0,
        cash_requested=0,
    ):
        self.from_player = from_player
        self.to_player = to_player
        self.offered_prop = offered_prop
        self.requested_prop = requested_prop
        self.cash_offered = cash_offered
        self.cash_requested = cash_requested

    def net_worth(self):
        po = self.offered_prop.price if self.offered_prop else 0
        pr = self.requested_prop.price if self.requested_prop else 0
        return (po + self.cash_offered) - (pr + self.cash_requested)


# Phase constants
PHASE_PRE_ROLL = "pre_roll"
PHASE_POST_ROLL = "post_roll"
PHASE_OUT_OF_TURN = "out_of_turn"
PHASE_AUCTION = "auction"


class MonopolyEnv:
    """
    Full 4-player Monopoly environment with correct turn sequencing.

    Turn structure per player:
      1. PRE_ROLL   : active player may build/trade/mortgage before rolling
      2. POST_ROLL  : active player rolls dice, lands, pays rent, buys property
      3. OUT_OF_TURN: other players may make trade offers (one round)
      Then move to next player's PRE_ROLL.
    """

    def __init__(self, agent_ids=None, max_rounds=200):
        self.agent_ids = agent_ids or [0]
        self.max_rounds = max_rounds
        self.reset()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def reset(self):
        self.players = [Player(i) for i in range(NUM_PLAYERS)]
        self.properties = {sq: Property(sq) for sq in PROPERTY_IDS}
        self.turn_order = list(range(NUM_PLAYERS))
        random.shuffle(self.turn_order)

        self.current_turn_idx = 0  # index into turn_order
        self.round = 0
        self.done = False
        self.pending_trades = {}  # sender_pid -> TradeOffer
        self.last_dice = (1, 1)

        # Phase tracking
        self.phase = PHASE_PRE_ROLL
        self.has_rolled = False  # has the active player rolled this turn?
        self.out_of_turn_pids = []  # which players still get out-of-turn actions
        self.consecutive_doubles = 0
        self.extra_roll_pending = False

        # Auction state. The landing player starts the bidding.
        self.auction_property_id = None
        self.auction_bidders = []
        self.auction_current_pid = None
        self.auction_high_bid = 0
        self.auction_high_bidder = None
        self.houses_available = HOUSE_SUPPLY
        self.hotels_available = HOTEL_SUPPLY

        # Only unpaid rent creates a creditor in the PPO-plus ruleset.
        self.debt_player = None
        self.debt_creditor = None
        self.debt_amount = 0

        # ── BUG 1 FIX: rescue flag ─────────────────────────────────────────
        # Set to a player's pid when they owe more than they have after paying
        # rent.  While set, get_allowed_actions() returns sell-house / mortgage
        # actions instead of the normal post-roll menu, giving the agent a
        # chance to raise funds before bankruptcy is declared.
        self.player_needs_funds = None

        self._update_monopolies()
        self._skip_bankrupt()
        return self._get_state(self.agent_ids[0])

    # ── Public API ─────────────────────────────────────────────────────────────

    def active_player_id(self) -> int:
        """The player whose turn it currently is (pre/post roll phases)."""
        return self.turn_order[self.current_turn_idx]

    def current_out_of_turn_player(self) -> Optional[int]:
        """During out-of-turn phase, which player acts next."""
        if self.phase == PHASE_OUT_OF_TURN and self.out_of_turn_pids:
            return self.out_of_turn_pids[0]
        return None

    def whose_turn(self) -> int:
        """
        Returns the player_id who should act RIGHT NOW.
        Callers use this to know who to ask for an action.
        """
        if self.phase == PHASE_OUT_OF_TURN:
            oot = self.current_out_of_turn_player()
            return oot if oot is not None else self.active_player_id()
        if self.phase == PHASE_AUCTION and self.auction_current_pid is not None:
            return self.auction_current_pid
        return self.active_player_id()

    def step(self, action_idx: int):
        """
        Apply action for whoever whose_turn() says should act.
        Returns (next_state, reward, done, info).
        """
        if self.done:
            return self._get_state(self.agent_ids[0]), 0.0, True, {}

        pid = self.whose_turn()
        player = self.players[pid]
        info = {"player": pid, "phase": self.phase}

        if player.bankrupt:
            self._advance_turn()
            return self._get_state(self.agent_ids[0]), 0.0, self.done, info

        allowed = self.get_allowed_actions(pid)
        if action_idx not in allowed:
            raise ValueError(
                f"Illegal action {action_idx} for player {pid} during "
                f"{self.phase}; allowed={allowed}"
            )

        self._apply_action(pid, action_idx, info)
        reward = self._compute_reward(self.agent_ids[0])
        self._check_game_over()

        return self._get_state(self.agent_ids[0]), reward, self.done, info

    def get_allowed_actions(self, pid: int = None) -> List[int]:
        """
        Return valid action indices for the given player RIGHT NOW.
        If pid is None, uses whose_turn().
        """
        if pid is None:
            pid = self.whose_turn()

        player = self.players[pid]
        active = self.active_player_id()
        allowed = []

        if player.bankrupt:
            return [int(ActionType.DO_NOTHING)]

        if self.phase == PHASE_AUCTION:
            if pid != self.auction_current_pid:
                return [int(ActionType.DO_NOTHING)]
            allowed = [int(AuctionAction.PASS)]
            for action, increment in AUCTION_ACTION_TO_INCREMENT.items():
                if self.auction_high_bid + increment <= player.cash:
                    allowed.append(int(action))
            return allowed

        # ── OUT-OF-TURN phase: non-active players ──────────────────────────
        if self.phase == PHASE_OUT_OF_TURN:
            if pid == self.current_out_of_turn_player():
                allowed.append(int(ActionType.END_TURN))  # skip out-of-turn
                # Can respond to incoming trade
                pending = self._incoming_trade(pid)
                if pending:
                    allowed.append(int(ActionType.ACCEPT_TRADE))
                    allowed.append(int(ActionType.DECLINE_TRADE))
                # Can make trade offers
                allowed += self._trade_offer_actions(pid)
                return allowed if allowed else [int(ActionType.END_TURN)]
            return [int(ActionType.DO_NOTHING)]

        # ── PRE-ROLL phase: active player before rolling ───────────────────
        if self.phase == PHASE_PRE_ROLL and pid == active:
            allowed.append(int(ActionType.END_TURN))  # end pre-roll, go to post-roll

            # Jail options
            if player.in_jail:
                if player.gooj_card:
                    allowed.append(int(ActionType.USE_GOOJ_CARD))
                if player.can_afford(JAIL_BAIL):
                    allowed.append(int(ActionType.PAY_BAIL))

            # Mortgage / unmortgage
            allowed += self._mortgage_actions(pid)
            allowed += self._sell_property_actions(pid)

            # Build / sell houses
            allowed += self._improve_actions(pid)

            # Trade offers
            allowed += self._trade_offer_actions(pid)

            # Respond to incoming trade
            pending = self._incoming_trade(pid)
            if pending:
                allowed.append(int(ActionType.ACCEPT_TRADE))
                allowed.append(int(ActionType.DECLINE_TRADE))

            return allowed if allowed else [int(ActionType.END_TURN)]

        # ── POST-ROLL phase: active player rolls then handles landing ──────
        if self.phase == PHASE_POST_ROLL and pid == active:
            if not self.has_rolled:
                # Must roll first
                if player.in_jail:
                    if player.gooj_card:
                        allowed.append(int(ActionType.USE_GOOJ_CARD))
                    if player.can_afford(JAIL_BAIL):
                        allowed.append(int(ActionType.PAY_BAIL))
                allowed.append(int(ActionType.ROLL_DICE))
                return allowed

            else:
                # Unpaid rent must be settled before the turn can continue.
                if self.debt_player == pid:
                    rescue = []
                    rescue += self._improve_actions(pid)  # sell_house / sell_hotel
                    rescue += self._mortgage_actions(pid)  # mortgage properties
                    rescue += self._sell_property_actions(pid)
                    if rescue:
                        return rescue
                    # Genuinely insolvent — nothing left to liquidate
                    return [int(ActionType.DECLARE_BANKRUPT)]

                # Normal post-roll menu
                sq = player.position
                prop = self.properties.get(sq)

                if prop and prop.owner is None and player.can_afford(prop.price):
                    allowed.append(int(ActionType.BUY_PROPERTY))

                # Can also mortgage to raise cash, or end turn
                allowed += self._mortgage_actions(pid)
                allowed += self._sell_property_actions(pid)
                allowed.append(int(ActionType.END_TURN))

                return allowed if allowed else [int(ActionType.END_TURN)]

        return [int(ActionType.DO_NOTHING)]

    # ── Action dispatch ────────────────────────────────────────────────────────

    def _apply_action(self, pid: int, action_idx: int, info: dict):
        player = self.players[pid]
        active = self.active_player_id()

        if self.phase == PHASE_AUCTION:
            self._handle_auction_action(pid, action_idx, info)
            return

        # ── Binary actions ─────────────────────────────────────────────────
        if action_idx < OFFSETS["mortgage"]:
            atype = ActionType(action_idx)

            if atype in (ActionType.DO_NOTHING,):
                pass  # no-op

            elif atype == ActionType.END_TURN:
                self._handle_end_turn(pid)

            elif atype == ActionType.ROLL_DICE:
                if self.phase == PHASE_POST_ROLL and not self.has_rolled:
                    self._do_roll(pid, info)

            elif atype == ActionType.BUY_PROPERTY:
                if self.phase == PHASE_POST_ROLL and self.has_rolled:
                    self._do_buy(pid)

            elif atype == ActionType.USE_GOOJ_CARD:
                if player.gooj_card and player.in_jail:
                    player.gooj_card = False
                    player.in_jail = False
                    player.jail_turns = 0

            elif atype == ActionType.PAY_BAIL:
                if player.in_jail and player.can_afford(JAIL_BAIL):
                    player.cash -= JAIL_BAIL
                    player.in_jail = False
                    player.jail_turns = 0

            elif atype == ActionType.DECLARE_BANKRUPT:
                self._do_bankrupt(pid)

            elif atype == ActionType.ACCEPT_TRADE:
                self._do_accept_trade(pid)

            elif atype == ActionType.DECLINE_TRADE:
                pending = self._incoming_trade_entry(pid)
                if pending is not None:
                    del self.pending_trades[pending[0]]

            return

        # ── Mortgage ───────────────────────────────────────────────────────
        if action_idx < OFFSETS["unmortgage"]:
            local = action_idx - OFFSETS["mortgage"]
            prop = self.properties[PROPERTY_IDS[local]]
            if prop.owner == pid and not prop.mortgaged and prop.houses == 0:
                prop.mortgaged = True
                player.cash += prop.mortgage_v
                self._settle_debt(pid)
            return

        # ── Unmortgage ─────────────────────────────────────────────────────
        if action_idx < OFFSETS["improve_house"]:
            local = action_idx - OFFSETS["unmortgage"]
            prop = self.properties[PROPERTY_IDS[local]]
            cost = int(prop.mortgage_v * 1.1)
            if prop.owner == pid and prop.mortgaged and player.can_afford(cost):
                prop.mortgaged = False
                player.cash -= cost
            return

        # ── Improve house ──────────────────────────────────────────────────
        if action_idx < OFFSETS["improve_hotel"]:
            local = action_idx - OFFSETS["improve_house"]
            prop = self.properties[REAL_ESTATE_IDS[local]]
            hp = prop.data["house_price"]
            if (
                prop.owner == pid
                and prop.is_monopoly
                and not prop.mortgaged
                and prop.houses < MAX_HOUSES
                and self.houses_available > 0
                and player.can_afford(hp)
            ):
                prop.houses += 1
                player.cash -= hp
                self.houses_available -= 1
            return

        # ── Improve hotel ──────────────────────────────────────────────────
        if action_idx < OFFSETS["sell_house"]:
            local = action_idx - OFFSETS["improve_hotel"]
            prop = self.properties[REAL_ESTATE_IDS[local]]
            hp = prop.data["house_price"]
            if (
                prop.owner == pid
                and prop.is_monopoly
                and not prop.mortgaged
                and prop.houses == MAX_HOUSES
                and self.hotels_available > 0
                and player.can_afford(hp)
            ):
                prop.houses = 5
                player.cash -= hp
                self.hotels_available -= 1
                self.houses_available += MAX_HOUSES
            return

        # ── Sell house ─────────────────────────────────────────────────────
        if action_idx < OFFSETS["sell_hotel"]:
            local = action_idx - OFFSETS["sell_house"]
            prop = self.properties[REAL_ESTATE_IDS[local]]
            if prop.owner == pid and 1 <= prop.houses <= MAX_HOUSES:
                prop.houses -= 1
                player.cash += prop.data["house_price"] // 2
                self.houses_available += 1
                self._settle_debt(pid)
            return

        # ── Sell hotel ─────────────────────────────────────────────────────
        if action_idx < OFFSETS["sell_prop"]:
            local = action_idx - OFFSETS["sell_hotel"]
            prop = self.properties[REAL_ESTATE_IDS[local]]
            if (
                prop.owner == pid
                and prop.houses == 5
                and self.houses_available >= MAX_HOUSES
            ):
                prop.houses = MAX_HOUSES
                player.cash += prop.data["house_price"] // 2
                self.hotels_available += 1
                self.houses_available -= MAX_HOUSES
                self._settle_debt(pid)
            return

        # ── Sell property to bank ──────────────────────────────────────────
        if action_idx < OFFSETS["buy_trade"]:
            local = action_idx - OFFSETS["sell_prop"]
            prop = self.properties[PROPERTY_IDS[local]]
            if prop.owner == pid and not prop.mortgaged and prop.houses == 0:
                player.cash += prop.mortgage_v
                player.properties.remove(prop)
                prop.owner = None
                prop.mortgaged = False
                self._update_monopolies()
                self._settle_debt(pid)
            return

        # ── Trade offers ───────────────────────────────────────────────────
        if action_idx < OFFSETS["sell_trade"]:
            self._make_trade_offer(pid, action_idx - OFFSETS["buy_trade"], "buy")
            return

        if action_idx < OFFSETS["exch_trade"]:
            self._make_trade_offer(pid, action_idx - OFFSETS["sell_trade"], "sell")
            return

        self._make_exchange_offer(pid, action_idx - OFFSETS["exch_trade"])

    # ── Turn / phase advancement ───────────────────────────────────────────────

    def _start_out_of_turn(self):
        active = self.active_player_id()
        self.phase = PHASE_OUT_OF_TURN
        self.out_of_turn_pids = [
            p
            for p in self.turn_order
            if p != active and not self.players[p].bankrupt
        ]
        self.consecutive_doubles = 0
        self.extra_roll_pending = False
        if not self.out_of_turn_pids:
            self._next_player()

    def _start_auction(self, property_id: int):
        active = self.active_player_id()
        start = self.turn_order.index(active)
        order = self.turn_order[start:] + self.turn_order[:start]
        self.phase = PHASE_AUCTION
        self.auction_property_id = property_id
        self.auction_bidders = [
            pid for pid in order if not self.players[pid].bankrupt
        ]
        self.auction_current_pid = self.auction_bidders[0]
        self.auction_high_bid = 0
        self.auction_high_bidder = None

    def _handle_auction_action(self, pid: int, action_idx: int, info: dict):
        if pid != self.auction_current_pid:
            return

        if action_idx == int(AuctionAction.PASS):
            self.auction_bidders.remove(pid)
            info["auction_pass"] = True
        elif action_idx in {int(a) for a in AUCTION_ACTION_TO_INCREMENT}:
            action = AuctionAction(action_idx)
            increment = AUCTION_ACTION_TO_INCREMENT[action]
            bid = self.auction_high_bid + increment
            if bid > self.players[pid].cash:
                return
            self.auction_high_bid = bid
            self.auction_high_bidder = pid
            info["auction_bid"] = bid
        else:
            return

        eligible = [
            bidder
            for bidder in self.auction_bidders
            if bidder != self.auction_high_bidder
        ]
        if not eligible:
            self._finish_auction(info)
            return

        start = self.turn_order.index(pid)
        for offset in range(1, len(self.turn_order) + 1):
            candidate = self.turn_order[(start + offset) % len(self.turn_order)]
            if candidate in eligible:
                self.auction_current_pid = candidate
                return

    def _finish_auction(self, info: dict):
        if self.auction_high_bidder is not None:
            winner = self.players[self.auction_high_bidder]
            prop = self.properties[self.auction_property_id]
            winner.cash -= self.auction_high_bid
            prop.owner = winner.player_id
            winner.properties.append(prop)
            self._update_monopolies()
            info["auction_winner"] = winner.player_id
            info["auction_price"] = self.auction_high_bid

        self.auction_property_id = None
        self.auction_bidders = []
        self.auction_current_pid = None
        self.auction_high_bid = 0
        self.auction_high_bidder = None

        player = self.players[self.active_player_id()]
        if self.extra_roll_pending and not player.in_jail:
            self.extra_roll_pending = False
            self.phase = PHASE_POST_ROLL
            self.has_rolled = False
        else:
            self._start_out_of_turn()

    def _handle_end_turn(self, pid: int):
        active = self.active_player_id()

        if self.phase == PHASE_PRE_ROLL and pid == active:
            # Move to post-roll: player now needs to roll
            self.phase = PHASE_POST_ROLL
            self.has_rolled = False

        elif self.phase == PHASE_POST_ROLL and pid == active:
            player = self.players[pid]
            prop = self.properties.get(player.position)
            if self.has_rolled and prop is not None and prop.owner is None:
                self._start_auction(prop.square_id)
            elif self.extra_roll_pending and not player.in_jail:
                self.extra_roll_pending = False
                self.phase = PHASE_POST_ROLL
                self.has_rolled = False
            else:
                self._start_out_of_turn()

        elif self.phase == PHASE_OUT_OF_TURN:
            # This out-of-turn player is done
            if pid in self.out_of_turn_pids:
                self.out_of_turn_pids.remove(pid)
            if not self.out_of_turn_pids:
                self._next_player()

    def _next_player(self):
        """Advance to the next non-bankrupt player's pre-roll phase."""
        n = len(self.turn_order)
        next_idx = (self.current_turn_idx + 1) % n
        skipped = 0
        while skipped < n:
            if not self.players[self.turn_order[next_idx]].bankrupt:
                break
            next_idx = (next_idx + 1) % n
            skipped += 1

        if next_idx <= self.current_turn_idx:
            self.round += 1

        self.current_turn_idx = next_idx
        self.phase = PHASE_PRE_ROLL
        self.has_rolled = False
        self.out_of_turn_pids = []
        self.pending_trades = {}
        self.consecutive_doubles = 0
        self.extra_roll_pending = False

    def _advance_turn(self):
        """Force-advance (used when a bankrupt player is encountered)."""
        self._next_player()

    def _skip_bankrupt(self):
        """Ensure current player is not bankrupt at game start."""
        for _ in range(NUM_PLAYERS):
            if not self.players[self.active_player_id()].bankrupt:
                break
            self.current_turn_idx = (self.current_turn_idx + 1) % NUM_PLAYERS

    # ── Core game mechanics ────────────────────────────────────────────────────

    def _do_roll(self, pid: int, info: dict):
        player = self.players[pid]
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        self.last_dice = (d1, d2)
        info["dice"] = (d1, d2)
        self.has_rolled = True

        # Jail handling. Doubles release the player but do not grant another roll.
        was_in_jail = player.in_jail
        if was_in_jail:
            player.jail_turns += 1
            if d1 == d2:
                player.in_jail = False
                player.jail_turns = 0
            elif player.jail_turns >= MAX_JAIL_TURNS:
                player.cash -= min(JAIL_BAIL, player.cash)
                player.in_jail = False
                player.jail_turns = 0
            else:
                # Stay in jail — turn ends
                return

        if not was_in_jail and d1 == d2:
            self.consecutive_doubles += 1
            if self.consecutive_doubles >= 3:
                player.position = JAIL_SQUARE
                player.in_jail = True
                player.jail_turns = 0
                self.consecutive_doubles = 0
                self.extra_roll_pending = False
                info["three_doubles"] = True
                return
            self.extra_roll_pending = True
        else:
            self.consecutive_doubles = 0
            self.extra_roll_pending = False

        # Move
        new_pos = (player.position + d1 + d2) % 40

        # Passed Go?
        if new_pos < player.position and not player.in_jail:
            player.cash += GO_SALARY

        player.position = new_pos
        self._handle_landing(pid, d1 + d2, info)

    def _handle_landing(self, pid: int, dice_total: int, info: dict):
        player = self.players[pid]
        sq = player.position
        info["landed_on"] = sq

        if sq == GO_TO_JAIL_SQUARE:
            player.position = JAIL_SQUARE
            player.in_jail = True
            player.jail_turns = 0
            self.consecutive_doubles = 0
            self.extra_roll_pending = False
            return

        if sq == INCOME_TAX_SQUARE:
            player.cash = max(0, player.cash - 200)
            return

        if sq == LUXURY_TAX_SQUARE:
            player.cash = max(0, player.cash - 100)
            return

        if sq not in self.properties:
            return  # Go, Jail, Free Parking, Chance, Community Chest

        prop = self.properties[sq]

        if prop.owner is None:
            info["can_buy"] = True
            return  # player decides whether to buy in get_allowed_actions

        if prop.owner == pid:
            return  # own property, no rent

        # Pay rent
        owner = self.players[prop.owner]
        n_rails = owner.railroads_owned()
        n_utils = owner.utilities_owned()
        rent = prop.get_rent(dice_total, n_rails, n_utils)

        owner_receives = min(rent, player.cash)
        player.cash -= owner_receives
        owner.cash += owner_receives
        info["rent_paid"] = owner_receives

        unpaid = rent - owner_receives
        if unpaid:
            self.debt_player = pid
            self.debt_creditor = owner.player_id
            self.debt_amount = unpaid
            self.player_needs_funds = pid

    def _do_buy(self, pid: int):
        player = self.players[pid]
        sq = player.position
        if sq not in self.properties:
            return
        prop = self.properties[sq]
        if prop.owner is None and player.can_afford(prop.price):
            prop.owner = pid
            player.cash -= prop.price
            player.properties.append(prop)
            self._update_monopolies()

    def _do_bankrupt(self, pid: int):
        player = self.players[pid]
        creditor_pid = self.debt_creditor if self.debt_player == pid else None

        # Return all improvements and pay their half-price proceeds first.
        for prop in player.properties:
            if prop.houses == 5:
                self.hotels_available += 1
                player.cash += 5 * (prop.data["house_price"] // 2)
            elif prop.houses > 0:
                self.houses_available += prop.houses
                player.cash += prop.houses * (prop.data["house_price"] // 2)
            prop.houses = 0
        self._settle_debt(pid)

        player.bankrupt = True
        creditor = (
            self.players[creditor_pid]
            if creditor_pid is not None and not self.players[creditor_pid].bankrupt
            else None
        )
        if creditor is not None:
            creditor.cash += player.cash
            creditor.gooj_card = creditor.gooj_card or player.gooj_card
            for prop in player.properties:
                prop.owner = creditor.player_id
                creditor.properties.append(prop)
        else:
            for prop in player.properties:
                prop.owner = None
                prop.mortgaged = False

        player.cash = 0
        player.gooj_card = False
        player.properties = []
        self._clear_debt(pid)
        self._update_monopolies()
        self._next_player()

    def _settle_debt(self, pid: int):
        if self.debt_player != pid or self.debt_amount <= 0:
            return
        player = self.players[pid]
        payment = min(player.cash, self.debt_amount)
        player.cash -= payment
        if self.debt_creditor is not None:
            self.players[self.debt_creditor].cash += payment
        self.debt_amount -= payment
        if self.debt_amount == 0:
            self._clear_debt(pid)

    def _clear_debt(self, pid: int):
        if self.debt_player != pid:
            return
        self.debt_player = None
        self.debt_creditor = None
        self.debt_amount = 0
        if self.player_needs_funds == pid:
            self.player_needs_funds = None

    def _do_accept_trade(self, pid: int):
        pending = self._incoming_trade_entry(pid)
        if pending is None:
            return
        sender, offer = pending
        del self.pending_trades[sender]

        s = self.players[sender]
        r = self.players[pid]

        valid = (
            offer.from_player == sender
            and offer.to_player == pid
            and not s.bankrupt
            and not r.bankrupt
            and offer.cash_offered >= 0
            and offer.cash_requested >= 0
            and s.can_afford(offer.cash_offered)
            and r.can_afford(offer.cash_requested)
            and (
                offer.offered_prop is None
                or (
                    offer.offered_prop.owner == sender
                    and offer.offered_prop.houses == 0
                    and offer.offered_prop in s.properties
                )
            )
            and (
                offer.requested_prop is None
                or (
                    offer.requested_prop.owner == pid
                    and offer.requested_prop.houses == 0
                    and offer.requested_prop in r.properties
                )
            )
        )
        if not valid:
            return

        s.cash -= offer.cash_offered
        r.cash += offer.cash_offered
        r.cash -= offer.cash_requested
        s.cash += offer.cash_requested

        if offer.offered_prop:
            offer.offered_prop.owner = pid
            s.properties.remove(offer.offered_prop)
            r.properties.append(offer.offered_prop)

        if offer.requested_prop:
            offer.requested_prop.owner = sender
            r.properties.remove(offer.requested_prop)
            s.properties.append(offer.requested_prop)

        self._update_monopolies()

    # ── Trade offer construction ───────────────────────────────────────────────

    def _make_trade_offer(self, pid: int, local_idx: int, mode: str):
        n_props = len(PROPERTY_IDS)
        n_cash = len(TRADE_CASH_LEVELS)
        player_idx = local_idx // (n_props * n_cash)
        rem = local_idx % (n_props * n_cash)
        prop_idx = rem // n_cash
        price_idx = rem % n_cash

        others = [i for i in range(NUM_PLAYERS) if i != pid]
        if player_idx >= len(others):
            return
        target_pid = others[player_idx]
        prop = self.properties[PROPERTY_IDS[prop_idx]]
        multiplier = TRADE_CASH_LEVELS[price_idx]
        cash_amount = int(prop.price * multiplier)

        if mode == "buy":
            if prop.owner != target_pid:
                return
            offer = TradeOffer(
                pid, target_pid, cash_offered=cash_amount, requested_prop=prop
            )
        else:
            if prop.owner != pid:
                return
            offer = TradeOffer(
                pid, target_pid, offered_prop=prop, cash_requested=cash_amount
            )

        self.pending_trades[pid] = offer

    def _make_exchange_offer(self, pid: int, local_idx: int):
        n_props = len(PROPERTY_IDS)
        player_idx = local_idx // (n_props * (n_props - 1))
        rem = local_idx % (n_props * (n_props - 1))
        offer_idx = rem // (n_props - 1)
        req_raw = rem % (n_props - 1)
        req_idx = req_raw if req_raw < offer_idx else req_raw + 1

        others = [i for i in range(NUM_PLAYERS) if i != pid]
        if player_idx >= len(others):
            return
        target_pid = others[player_idx]
        offered_prop = self.properties[PROPERTY_IDS[offer_idx]]
        req_prop = self.properties[PROPERTY_IDS[req_idx]]

        if offered_prop.owner != pid or req_prop.owner != target_pid:
            return
        if offered_prop.houses > 0 or req_prop.houses > 0:
            return

        self.pending_trades[pid] = TradeOffer(
            pid, target_pid, offered_prop=offered_prop, requested_prop=req_prop
        )

    # ── Helpers for allowed actions ────────────────────────────────────────────

    def _mortgage_actions(self, pid: int) -> List[int]:
        player = self.players[pid]
        allowed = []
        for i, sq in enumerate(PROPERTY_IDS):
            prop = self.properties[sq]
            if prop.owner == pid:
                if not prop.mortgaged and prop.houses == 0:
                    allowed.append(OFFSETS["mortgage"] + i)
                cost = int(prop.mortgage_v * 1.1)
                if prop.mortgaged and player.can_afford(cost):
                    allowed.append(OFFSETS["unmortgage"] + i)
        return allowed

    def _improve_actions(self, pid: int) -> List[int]:
        player = self.players[pid]
        allowed = []
        for i, sq in enumerate(REAL_ESTATE_IDS):
            prop = self.properties[sq]
            if prop.owner != pid:
                continue
            hp = prop.data["house_price"]
            if (
                prop.is_monopoly
                and not prop.mortgaged
                and prop.houses < MAX_HOUSES
                and self.houses_available > 0
                and player.can_afford(hp)
            ):
                allowed.append(OFFSETS["improve_house"] + i)
            if (
                prop.is_monopoly
                and not prop.mortgaged
                and prop.houses == MAX_HOUSES
                and self.hotels_available > 0
                and player.can_afford(hp)
            ):
                allowed.append(OFFSETS["improve_hotel"] + i)
            if 1 <= prop.houses < 5:
                allowed.append(OFFSETS["sell_house"] + i)
            if prop.houses == 5 and self.houses_available >= MAX_HOUSES:
                allowed.append(OFFSETS["sell_hotel"] + i)
        return allowed

    def _sell_property_actions(self, pid: int) -> List[int]:
        return [
            OFFSETS["sell_prop"] + i
            for i, sq in enumerate(PROPERTY_IDS)
            if self.properties[sq].owner == pid
            and not self.properties[sq].mortgaged
            and self.properties[sq].houses == 0
        ]

    def _trade_offer_actions(self, pid: int) -> List[int]:
        """Only return trade actions for properties that actually exist and are owned."""
        if pid in self.pending_trades:
            return []
        allowed = []
        player = self.players[pid]
        all_others = [i for i in range(NUM_PLAYERS) if i != pid]
        if self.phase == PHASE_OUT_OF_TURN:
            current = self.out_of_turn_pids.index(pid)
            future_responders = set(self.out_of_turn_pids[current + 1 :])
        else:
            future_responders = set(all_others)
        targets = [
            target
            for target in all_others
            if target in future_responders and not self.players[target].bankrupt
        ]
        stride = len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)

        for target_pid in targets:
            t_idx = all_others.index(target_pid)
            target = self.players[target_pid]
            for i, sq in enumerate(PROPERTY_IDS):
                prop = self.properties[sq]
                # Buy offer: target owns it, we want it
                if prop.owner == target_pid and prop.houses == 0:
                    for j, multiplier in enumerate(TRADE_CASH_LEVELS):
                        if not player.can_afford(int(prop.price * multiplier)):
                            continue
                        allowed.append(
                            OFFSETS["buy_trade"]
                            + t_idx * stride
                            + i * len(TRADE_CASH_LEVELS)
                            + j
                        )
                # Sell offer: we own it
                if prop.owner == pid and prop.houses == 0:
                    for j, multiplier in enumerate(TRADE_CASH_LEVELS):
                        if not target.can_afford(int(prop.price * multiplier)):
                            continue
                        allowed.append(
                            OFFSETS["sell_trade"]
                            + t_idx * stride
                            + i * len(TRADE_CASH_LEVELS)
                            + j
                        )

            for offer_idx, offer_sq in enumerate(PROPERTY_IDS):
                offered = self.properties[offer_sq]
                if offered.owner != pid or offered.houses > 0:
                    continue
                for request_idx, request_sq in enumerate(PROPERTY_IDS):
                    requested = self.properties[request_sq]
                    if requested.owner != target_pid or requested.houses > 0:
                        continue
                    request_raw = (
                        request_idx if request_idx < offer_idx else request_idx - 1
                    )
                    allowed.append(
                        OFFSETS["exch_trade"]
                        + t_idx * len(PROPERTY_IDS) * (len(PROPERTY_IDS) - 1)
                        + offer_idx * (len(PROPERTY_IDS) - 1)
                        + request_raw
                    )
        return allowed

    def _incoming_trade_entry(self, pid: int) -> Optional[tuple[int, TradeOffer]]:
        start = self.turn_order.index(pid)
        for offset in range(1, len(self.turn_order)):
            sender = self.turn_order[(start + offset) % len(self.turn_order)]
            offer = self.pending_trades.get(sender)
            if offer is not None and offer.to_player == pid:
                return sender, offer
        return None

    def _incoming_trade(self, pid: int) -> Optional[TradeOffer]:
        pending = self._incoming_trade_entry(pid)
        return None if pending is None else pending[1]

    # ── Reward & game-over ─────────────────────────────────────────────────────

    def _update_monopolies(self):
        for color, squares in COLOR_GROUPS.items():
            owners = [self.properties[s].owner for s in squares]
            is_mono = len(set(owners)) == 1 and owners[0] is not None
            for s in squares:
                self.properties[s].is_monopoly = is_mono

    def _compute_reward(self, pid: int) -> float:
        """Bounded net-worth potential used for decision-to-decision shaping."""
        if self.players[pid].bankrupt:
            return -1.0
        active = [p for p in self.players if not p.bankrupt]
        if len(active) <= 1:
            return 1.0

        own = self.players[pid].net_worth()
        others = [
            player.net_worth()
            for player in active
            if player.player_id != pid
        ]
        mean_other = float(np.mean(others))
        return float(
            np.clip(
                (own - mean_other) / (abs(own) + abs(mean_other) + 1e-9),
                -1.0,
                1.0,
            )
        )

    def _check_game_over(self):
        active = [p for p in self.players if not p.bankrupt]
        if len(active) <= 1 or self.round >= self.max_rounds:
            self.done = True

    def _get_state(self, pid: int) -> np.ndarray:
        return build_state_vector(self.players, self.properties, pid, self)

    def winner(self) -> Optional[int]:
        active = [p for p in self.players if not p.bankrupt]
        if len(active) == 1:
            return active[0].player_id
        return max(self.players, key=lambda p: p.net_worth()).player_id
