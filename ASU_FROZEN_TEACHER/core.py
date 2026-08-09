"""Frozen ASU-inspired value and rollout teachers for ``ppo-plus-v2``."""

from __future__ import annotations

import copy
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from itertools import count
from typing import Iterator, Mapping

from monopoly_game_engine.actions import (  # noqa: E402
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    PROPERTY_IDS,
    ActionType,
    AuctionAction,
    action_to_description,
)
from monopoly_game_engine.constants import (  # noqa: E402
    COLOR_GROUPS,
    GO_TO_JAIL_SQUARE,
    MAX_HOUSES,
    NUM_PLAYERS,
    PROPERTIES,
    RULESET_VERSION,
)
from monopoly_game_engine.env import PHASE_AUCTION  # noqa: E402

from .spec import (  # noqa: E402
    ASU_ROLLOUT_V1,
    ASU_VALUE_V1,
    FROZEN_SPEC_FINGERPRINT,
    FROZEN_SPEC_HASH,
)
from .types import (  # noqa: E402
    CandidateScore,
    Decision,
    RentProjection,
    SafetyBreakdown,
    SafetyRejection,
    ValueBreakdown,
)


SHORT_TURNS = 5
LONG_LAPS = 5
MINIMUM_CASH = 200
TERMINAL_UTILITY = 1_000_000.0
ROLLOUT_SEED = 0
ROLLOUT_SHORTLIST = 8
ROLLOUTS_PER_ACTION = 8
ROLLOUT_DECISIONS = 32
_EPSILON = 1e-12
_TURN_STATE = tuple[int, bool, int]
_LANDING = tuple[int, int]


@contextmanager
def preserve_global_rng() -> Iterator[None]:
    """Restore Python, NumPy, and already-imported Torch RNGs on exit."""

    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_states = None
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy is not None and numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


class _PrivateGame:
    """An environment paired with a private stdlib random stream."""

    __slots__ = ("env", "random_state")

    def __init__(self, env, seed: int):
        self.env = copy.deepcopy(env)
        self.random_state = random.Random(seed).getstate()

    def step(self, action: int):
        outer_state = random.getstate()
        try:
            random.setstate(self.random_state)
            result = self.env.step(action)
            self.random_state = random.getstate()
            return result
        finally:
            random.setstate(outer_state)


def _add_scaled(target: dict, source: Mapping, scale: Fraction) -> None:
    for key, value in source.items():
        target[key] += scale * value


def _frozen_counter(counter: Mapping) -> tuple:
    return tuple(sorted(counter.items()))


def _landing_state(position: int) -> _TURN_STATE:
    if position == GO_TO_JAIL_SQUARE:
        return (10, True, 0)
    return (position, False, 0)


@lru_cache(maxsize=None)
def _free_turn(
    position: int, doubles: int
) -> tuple[
    tuple[tuple[_TURN_STATE, Fraction], ...], tuple[tuple[_LANDING, Fraction], ...]
]:
    transitions: dict[_TURN_STATE, Fraction] = defaultdict(Fraction)
    landings: dict[_LANDING, Fraction] = defaultdict(Fraction)
    die_probability = Fraction(1, 36)
    for first in range(1, 7):
        for second in range(1, 7):
            total = first + second
            is_double = first == second
            if is_double and doubles + 1 >= 3:
                transitions[(10, True, 0)] += die_probability
                continue
            destination = (position + total) % 40
            landings[(destination, total)] += die_probability
            state = _landing_state(destination)
            if state[1] or not is_double:
                transitions[state] += die_probability
                continue
            later_states, later_landings = _free_turn(destination, doubles + 1)
            _add_scaled(transitions, dict(later_states), die_probability)
            _add_scaled(landings, dict(later_landings), die_probability)
    return _frozen_counter(transitions), _frozen_counter(landings)


@lru_cache(maxsize=None)
def _complete_turn(
    position: int, in_jail: bool, jail_turns: int
) -> tuple[
    tuple[tuple[_TURN_STATE, Fraction], ...], tuple[tuple[_LANDING, Fraction], ...]
]:
    if not in_jail:
        return _free_turn(position, 0)

    transitions: dict[_TURN_STATE, Fraction] = defaultdict(Fraction)
    landings: dict[_LANDING, Fraction] = defaultdict(Fraction)
    die_probability = Fraction(1, 36)
    for first in range(1, 7):
        for second in range(1, 7):
            total = first + second
            if first != second and jail_turns + 1 < 3:
                transitions[(10, True, jail_turns + 1)] += die_probability
                continue
            destination = (position + total) % 40
            landings[(destination, total)] += die_probability
            transitions[_landing_state(destination)] += die_probability
    return _frozen_counter(transitions), _frozen_counter(landings)


@lru_cache(maxsize=None)
def _project_landings(
    position: int, in_jail: bool, jail_turns: int, turns: int
) -> tuple[tuple[_LANDING, Fraction], ...]:
    states: dict[_TURN_STATE, Fraction] = {
        (int(position), bool(in_jail), int(jail_turns)): Fraction(1)
    }
    landings: dict[_LANDING, Fraction] = defaultdict(Fraction)
    for _ in range(turns):
        next_states: dict[_TURN_STATE, Fraction] = defaultdict(Fraction)
        for state, state_probability in states.items():
            transitions, turn_landings = _complete_turn(*state)
            _add_scaled(next_states, dict(transitions), state_probability)
            _add_scaled(landings, dict(turn_landings), state_probability)
        states = next_states
    return _frozen_counter(landings)


@lru_cache(maxsize=None)
def _expected_landings_float(
    position: int, in_jail: bool, jail_turns: int, turns: int
) -> tuple[tuple[_LANDING, float], ...]:
    return tuple(
        (landing, float(probability))
        for landing, probability in _project_landings(
            position, in_jail, jail_turns, turns
        )
    )


def expected_landings(player, turns: int = SHORT_TURNS) -> dict[_LANDING, float]:
    """Expected deed landings keyed by ``(square, dice_total)``."""

    if turns < 1:
        raise ValueError("turns must be positive")
    return dict(
        _expected_landings_float(
            player.position,
            player.in_jail,
            player.jail_turns,
            turns,
        )
    )


def movement_probabilities(player, turns: int = SHORT_TURNS) -> dict[int, float]:
    """Aggregate expected landing counts by board square."""

    result: dict[int, float] = defaultdict(float)
    for (square, _dice_total), probability in expected_landings(player, turns).items():
        result[square] += probability
    return dict(sorted(result.items()))


def _owned_count(env, owner: int, color: str) -> int:
    return sum(
        prop.owner == owner and prop.color == color for prop in env.properties.values()
    )


def deed_rent(env, square: int, dice_total: int = 7) -> int:
    """Current ppo-plus-v2 rent, including exact railroad/utility scaling."""

    prop = env.properties[square]
    if prop.owner is None or prop.mortgaged:
        return 0
    owner = prop.owner
    if prop.color == "railroad":
        return prop.data["rent"][min(_owned_count(env, owner, "railroad") - 1, 3)]
    if prop.color == "utility":
        utilities = _owned_count(env, owner, "utility")
        return prop.data["rent"][0 if utilities == 1 else 1] * dice_total
    if prop.houses:
        return prop.data["rent"][min(prop.houses, 5)]
    group = COLOR_GROUPS[prop.color]
    monopoly = all(env.properties[item].owner == owner for item in group)
    return prop.data["rent"][0] * (2 if monopoly else 1)


def rent_projection(env, player_id: int, turns: int = SHORT_TURNS) -> RentProjection:
    """Exact expected incoming and outgoing rent over complete turns."""

    root = env.players[player_id]
    root_landings = expected_landings(root, turns)
    income = 0.0
    exposure = 0.0
    worst = 0.0

    for (square, dice_total), probability in root_landings.items():
        prop = env.properties.get(square)
        if prop is None or prop.owner in (None, player_id):
            continue
        rent = deed_rent(env, square, dice_total)
        exposure += probability * rent
        if probability > 0:
            worst = max(worst, float(rent))

    for opponent in env.players:
        if opponent.player_id == player_id or opponent.bankrupt:
            continue
        for (square, dice_total), probability in expected_landings(
            opponent, turns
        ).items():
            prop = env.properties.get(square)
            if prop is not None and prop.owner == player_id:
                income += probability * deed_rent(env, square, dice_total)

    return RentProjection(
        income=float(income),
        exposure=float(exposure),
        net=float(income - exposure),
        worst_reachable_rent=float(worst),
    )


def long_rent_projection(env, player_id: int) -> RentProjection:
    """Five-lap rent projection with uniform deed landings."""

    visits_per_deed = Fraction(LONG_LAPS * 40, 7 * len(PROPERTY_IDS))
    root = env.players[player_id]
    if root.bankrupt:
        return RentProjection(0.0, 0.0, 0.0, 0.0)
    opponents = [
        player
        for player in env.players
        if player.player_id != player_id and not player.bankrupt
    ]
    income = sum(
        float(visits_per_deed) * deed_rent(env, square, 7) * len(opponents)
        for square, prop in env.properties.items()
        if prop.owner == player_id
    )
    exposure = sum(
        float(visits_per_deed) * deed_rent(env, square, 7)
        for square, prop in env.properties.items()
        if prop.owner is not None and prop.owner != player_id
    )
    worst = max(
        (
            float(deed_rent(env, square, 12))
            for square, prop in env.properties.items()
            if prop.owner is not None and prop.owner != player_id
        ),
        default=0.0,
    )
    return RentProjection(income, exposure, income - exposure, worst)


def asset_value(env, player_id: int) -> float:
    """List price plus development spend, excluding cash and mortgages."""

    value = 0.0
    for prop in env.properties.values():
        if prop.owner != player_id or prop.mortgaged:
            continue
        value += prop.price
        if prop.is_real_estate:
            value += prop.houses * prop.data["house_price"]
    return value


def liquidatable_worth(env, player_id: int) -> float:
    """Maximum additional cash reachable through current liquidation rules."""

    properties = [prop for prop in env.properties.values() if prop.owner == player_id]
    houses_returnable = env.houses_available + sum(
        prop.houses for prop in properties if 1 <= prop.houses <= MAX_HOUSES
    )
    hotels_can_be_broken = houses_returnable >= MAX_HOUSES
    worth = 0.0
    for prop in properties:
        can_clear = prop.houses != 5 or hotels_can_be_broken
        if prop.houses and can_clear:
            worth += prop.houses * (prop.data["house_price"] // 2)
        if not prop.mortgaged and can_clear:
            worth += prop.mortgage_v
    return worth


def _hypothetical_group_rent(
    color: str,
    squares: tuple[int, ...],
    levels: tuple[int, ...],
    enabled: tuple[bool, ...],
) -> float:
    if color == "railroad":
        rent = PROPERTIES[squares[0]]["rent"][min(len(squares) - 1, 3)]
        return float(sum(rent for active in enabled if active))
    if color == "utility":
        multiplier = PROPERTIES[squares[0]]["rent"][1 if len(squares) > 1 else 0]
        return float(sum(multiplier * 7 for active in enabled if active))
    total = 0.0
    for square, level, active in zip(squares, levels, enabled):
        if not active:
            continue
        rents = PROPERTIES[square]["rent"]
        total += rents[min(level, 5)] if level else rents[0] * 2
    return total


def _max_developed_rent(
    color: str,
    squares: tuple[int, ...],
    levels: tuple[int, ...],
    enabled: tuple[bool, ...],
    budget: float,
    houses_available: int,
    hotels_available: int,
) -> float:
    if color in ("railroad", "utility"):
        return _hypothetical_group_rent(color, squares, levels, enabled)

    stack = [(levels, houses_available, hotels_available, 0)]
    least_spend: dict[tuple[tuple[int, ...], int, int], int] = {}
    best = _hypothetical_group_rent(color, squares, levels, enabled)
    while stack:
        current, houses, hotels, spent = stack.pop()
        state_key = (current, houses, hotels)
        if least_spend.get(state_key, spent + 1) <= spent:
            continue
        least_spend[state_key] = spent
        best = max(
            best,
            _hypothetical_group_rent(color, squares, current, enabled),
        )
        for index, (square, active) in enumerate(zip(squares, enabled)):
            if not active:
                continue
            price = PROPERTIES[square]["house_price"]
            if spent + price > budget + _EPSILON:
                continue
            level = current[index]
            if level < MAX_HOUSES and houses > 0:
                changed = list(current)
                changed[index] += 1
                stack.append((tuple(changed), houses - 1, hotels, spent + price))
            elif level == MAX_HOUSES and hotels > 0:
                changed = list(current)
                changed[index] = 5
                stack.append(
                    (
                        tuple(changed),
                        houses + MAX_HOUSES,
                        hotels - 1,
                        spent + price,
                    )
                )
    return best


def monopoly_value(env, player_id: int, r_long: float | None = None) -> float:
    """Maximum discounted potential rent for a color the player has entered."""

    if r_long is None:
        r_long = long_rent_projection(env, player_id).net
    funds = env.players[player_id].cash + LONG_LAPS * 200 + r_long
    best = 0.0
    for color, group in COLOR_GROUPS.items():
        squares = tuple(group)
        owned = [
            square for square in squares if env.properties[square].owner == player_id
        ]
        if not owned:
            continue
        missing = len(squares) - len(owned)
        acquisition_cost = sum(
            env.properties[square].price
            for square in squares
            if env.properties[square].owner != player_id
        )
        remaining = funds - acquisition_cost
        if remaining < -_EPSILON:
            continue

        levels = tuple(
            env.properties[square].houses
            if env.properties[square].owner == player_id
            else 0
            for square in squares
        )
        mortgaged_indices = [
            index
            for index, square in enumerate(squares)
            if env.properties[square].owner == player_id
            and env.properties[square].mortgaged
        ]
        always_enabled = [
            env.properties[square].owner != player_id
            or not env.properties[square].mortgaged
            for square in squares
        ]
        group_best = 0.0
        for mask in range(1 << len(mortgaged_indices)):
            enabled = list(always_enabled)
            unmortgage_cost = 0
            for bit, index in enumerate(mortgaged_indices):
                if mask & (1 << bit):
                    enabled[index] = True
                    unmortgage_cost += int(
                        env.properties[squares[index]].mortgage_v * 1.1
                    )
            budget = remaining - unmortgage_cost
            if budget < -_EPSILON:
                continue
            group_best = max(
                group_best,
                _max_developed_rent(
                    color,
                    squares,
                    levels,
                    tuple(enabled),
                    budget,
                    env.houses_available,
                    env.hotels_available,
                ),
            )
        best = max(best, group_best / (2**missing))
    return best


def evaluate_value(env, player_id: int) -> ValueBreakdown:
    """Evaluate one player without mutating the environment."""

    player = env.players[player_id]
    if player.bankrupt:
        return ValueBreakdown(0.0, 0.0, 0.0, 0.0, -TERMINAL_UTILITY, -TERMINAL_UTILITY)
    if env.done:
        terminal = TERMINAL_UTILITY if env.winner() == player_id else -TERMINAL_UTILITY
        return ValueBreakdown(0.0, 0.0, 0.0, 0.0, terminal, terminal)

    assets = asset_value(env, player_id)
    short = rent_projection(env, player_id, SHORT_TURNS).net
    long = long_rent_projection(env, player_id).net
    monopoly = monopoly_value(env, player_id, long)
    total = assets + short + long + monopoly
    return ValueBreakdown(assets, short, long, monopoly, 0.0, total)


def safety_breakdown(env, player_id: int) -> SafetyBreakdown:
    one_round = rent_projection(env, player_id, 1)
    cash = float(env.players[player_id].cash)
    liquidation = liquidatable_worth(env, player_id)
    first = cash + one_round.net - MINIMUM_CASH
    second = cash + one_round.income + liquidation - one_round.worst_reachable_rent
    return SafetyBreakdown(
        cash_after=cash,
        next_round_net_rent=one_round.net,
        next_round_rent_income=one_round.income,
        liquidatable_worth=liquidation,
        worst_reachable_rent=one_round.worst_reachable_rent,
        cash_floor_margin=first,
        solvency_margin=second,
        passed=first >= 0 and second > 0,
    )


def _average_values(values: list[ValueBreakdown]) -> ValueBreakdown:
    count_values = float(len(values))
    fields = (
        "m_assets",
        "r_short",
        "r_long",
        "m_monopoly",
        "terminal_utility",
        "total",
    )
    averaged = [
        sum(getattr(value, field) for value in values) / count_values
        for field in fields
    ]
    return ValueBreakdown(*averaged)


def _average_safety(values: list[SafetyBreakdown]) -> SafetyBreakdown:
    count_values = float(len(values))
    fields = (
        "cash_after",
        "next_round_net_rent",
        "next_round_rent_income",
        "liquidatable_worth",
        "worst_reachable_rent",
        "cash_floor_margin",
        "solvency_margin",
    )
    averaged = [
        sum(getattr(value, field) for value in values) / count_values
        for field in fields
    ]
    return SafetyBreakdown(
        *averaged,
        passed=averaged[-2] >= 0 and averaged[-1] > 0,
    )


@lru_cache(maxsize=1)
def _dice_seeds() -> dict[tuple[int, int], int]:
    seeds: dict[tuple[int, int], int] = {}
    for seed in count():
        rng = random.Random(seed)
        pair = (rng.randint(1, 6), rng.randint(1, 6))
        seeds.setdefault(pair, seed)
        if len(seeds) == 36:
            return seeds
    raise AssertionError("unreachable")


def _is_trade_offer(action: int) -> bool:
    return OFFSETS["buy_trade"] <= action < OFFSETS["auction"]


def _is_auction_bid(action: int) -> bool:
    return action in {int(item) for item in AUCTION_ACTION_TO_INCREMENT}


def _is_discretionary(action: int) -> bool:
    if action in (
        int(ActionType.BUY_PROPERTY),
        int(ActionType.PAY_BAIL),
        int(ActionType.ACCEPT_TRADE),
    ):
        return True
    if _is_trade_offer(action) or _is_auction_bid(action):
        return True
    return OFFSETS["unmortgage"] <= action < OFFSETS["sell_house"]


def _is_progress_fallback(action: int) -> bool:
    return action in (
        int(ActionType.DO_NOTHING),
        int(ActionType.END_TURN),
        int(ActionType.ROLL_DICE),
        int(ActionType.USE_GOOJ_CARD),
        int(ActionType.DECLINE_TRADE),
        int(ActionType.DECLARE_BANKRUPT),
        int(AuctionAction.PASS),
    )


def semantic_priority(action: int) -> int:
    """Frozen action-family tie priority; lower is preferred."""

    if action == int(ActionType.ACCEPT_TRADE):
        return 10
    if action == int(ActionType.BUY_PROPERTY):
        return 20
    if OFFSETS["improve_hotel"] <= action < OFFSETS["sell_house"]:
        return 30
    if OFFSETS["improve_house"] <= action < OFFSETS["improve_hotel"]:
        return 31
    if OFFSETS["unmortgage"] <= action < OFFSETS["improve_house"]:
        return 40
    if action == int(ActionType.USE_GOOJ_CARD):
        return 50
    if action == int(ActionType.PAY_BAIL):
        return 51
    if action == int(ActionType.ROLL_DICE):
        return 52
    if _is_trade_offer(action):
        return 60
    if OFFSETS["mortgage"] <= action < OFFSETS["unmortgage"]:
        return 70
    if OFFSETS["sell_house"] <= action < OFFSETS["buy_trade"]:
        return 71
    if action == int(ActionType.DECLINE_TRADE):
        return 80
    if _is_auction_bid(action):
        return 90
    if action == int(AuctionAction.PASS):
        return 91
    if action == int(ActionType.END_TURN):
        return 100
    if action == int(ActionType.DO_NOTHING):
        return 110
    if action == int(ActionType.DECLARE_BANKRUPT):
        return 120
    return 200


def _safety_reasons(safety: SafetyBreakdown) -> tuple[str, ...]:
    reasons = []
    if safety.cash_floor_margin < 0:
        reasons.append("cash_after + next_round_net_rent < 200")
    if safety.solvency_margin <= 0:
        reasons.append(
            "cash_after + next_round_rent_income + liquidatable_worth - worst_reachable_rent <= 0"
        )
    return tuple(reasons)


def _candidate_sort_key(candidate: CandidateScore) -> tuple[float, int, int]:
    return (-candidate.value.total, candidate.semantic_priority, candidate.action)


class ASUValueV1:
    """Deterministic one-step ASU-inspired reconstructed teacher."""

    policy_id = ASU_VALUE_V1

    def __init__(self, player_id: int):
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        self.player_id = player_id

    def value(self, env, player_id: int | None = None) -> ValueBreakdown:
        return evaluate_value(env, self.player_id if player_id is None else player_id)

    def safety(self, env, player_id: int | None = None) -> SafetyBreakdown:
        return safety_breakdown(env, self.player_id if player_id is None else player_id)

    def choose_action(self, env) -> int:
        return self.decide(env).selected_action

    def _step_copy(self, env, action: int, seed: int = 0):
        game = _PrivateGame(env, seed)
        game.step(action)
        return game.env

    def _roll_outcome(self, env, action: int) -> tuple[ValueBreakdown, SafetyBreakdown]:
        values = []
        safety = []
        for pair, seed in sorted(_dice_seeds().items()):
            rolled = self._step_copy(env, action, seed)
            if tuple(rolled.last_dice) != pair:
                raise AssertionError("dice seed no longer produces its frozen outcome")
            values.append(self.value(rolled))
            safety.append(self.safety(rolled))
        return _average_values(values), _average_safety(safety)

    def _trade_candidate(
        self,
        env,
        action: int,
        forced: bool,
        mandatory: bool,
        before_values: Mapping[int, ValueBreakdown],
    ) -> CandidateScore:
        root = self.player_id
        incoming = (
            env._incoming_trade(root)
            if action == int(ActionType.ACCEPT_TRADE)
            else None
        )
        if incoming is not None:
            proposer, recipient = incoming.from_player, root
            after = self._step_copy(env, action)
        else:
            after = self._step_copy(env, action)
            offer = after.pending_trades.get(root)
            if offer is None:
                raise RuntimeError(
                    f"legal trade action {action} did not create an offer"
                )
            proposer, recipient = root, offer.to_player
            after._do_accept_trade(recipient)
            after._check_game_over()

        after_values = {
            party: self.value(after, party) for party in {proposer, recipient}
        }
        proposer_gain = after_values[proposer].total - before_values[proposer].total
        recipient_gain = after_values[recipient].total - before_values[recipient].total
        root_value = after_values[root]
        after_safety = {
            party: self.safety(after, party) for party in {proposer, recipient}
        }
        root_safety = after_safety[root]
        reasons: list[str] = []
        if proposer_gain <= 0:
            reasons.append("proposer_gain <= 0")
        if recipient_gain < 0:
            reasons.append("recipient_gain < 0")
        for role, party in (("proposer", proposer), ("recipient", recipient)):
            for reason in _safety_reasons(after_safety[party]):
                reasons.append(f"{role}: {reason}")
        if forced:
            reasons = []
        return CandidateScore(
            action=action,
            description=action_to_description(action),
            value=root_value,
            safety=root_safety,
            eligible=not reasons,
            mandatory=mandatory,
            forced=forced,
            semantic_priority=semantic_priority(action),
            rejection_reasons=tuple(reasons),
            proposer_gain=proposer_gain,
            recipient_gain=recipient_gain,
        )

    def _auction_ceiling(self, env) -> float:
        root = self.player_id
        square = env.auction_property_id
        if square is None:
            return 0.0
        baseline = self.value(env).total
        acquired = copy.deepcopy(env)
        prop = acquired.properties[square]
        if prop.owner is None:
            prop.owner = root
            acquired.players[root].properties.append(prop)
            acquired._update_monopolies()
        return max(0.0, self.value(acquired).total - baseline)

    def _auction_candidate(
        self,
        env,
        action: int,
        forced: bool,
        mandatory: bool,
        ceiling: float,
    ) -> CandidateScore:
        if action == int(AuctionAction.PASS):
            after = self._step_copy(env, action)
            value = self.value(after)
            safety = self.safety(after)
            reasons: tuple[str, ...] = ()
        else:
            bid = (
                env.auction_high_bid
                + AUCTION_ACTION_TO_INCREMENT[AuctionAction(action)]
            )
            after = copy.deepcopy(env)
            prop = after.properties[after.auction_property_id]
            prop.owner = self.player_id
            after.players[self.player_id].properties.append(prop)
            after.players[self.player_id].cash -= bid
            after._update_monopolies()
            value = self.value(after)
            safety = self.safety(after)
            collected = list(_safety_reasons(safety))
            if bid > ceiling + _EPSILON:
                collected.append("total bid exceeds marginal ASU auction ceiling")
            reasons = () if forced else tuple(collected)
        return CandidateScore(
            action=action,
            description=action_to_description(action),
            value=value,
            safety=safety,
            eligible=not reasons,
            mandatory=mandatory,
            forced=forced,
            semantic_priority=semantic_priority(action),
            rejection_reasons=reasons,
            auction_ceiling=ceiling,
        )

    def _ordinary_candidate(
        self,
        env,
        action: int,
        forced: bool,
        mandatory: bool,
    ) -> CandidateScore:
        if action == int(ActionType.ROLL_DICE):
            value, safety = self._roll_outcome(env, action)
        else:
            after = self._step_copy(env, action)
            value = self.value(after)
            safety = self.safety(after)
        reasons = (
            () if forced or not _is_discretionary(action) else _safety_reasons(safety)
        )
        return CandidateScore(
            action=action,
            description=action_to_description(action),
            value=value,
            safety=safety,
            eligible=not reasons,
            mandatory=mandatory,
            forced=forced,
            semantic_priority=semantic_priority(action),
            rejection_reasons=reasons,
        )

    @staticmethod
    def _select(candidates: tuple[CandidateScore, ...], phase: str) -> int:
        eligible = [candidate for candidate in candidates if candidate.eligible]
        if not eligible:
            eligible = list(candidates)

        if phase == PHASE_AUCTION:
            bids = [
                candidate for candidate in eligible if _is_auction_bid(candidate.action)
            ]
            if bids:
                return max(
                    bids,
                    key=lambda candidate: (
                        AUCTION_ACTION_TO_INCREMENT[AuctionAction(candidate.action)],
                        -candidate.action,
                    ),
                ).action
            passes = [
                candidate
                for candidate in candidates
                if candidate.action == int(AuctionAction.PASS)
            ]
            if passes:
                return passes[0].action

        discretionary = [
            candidate for candidate in candidates if _is_discretionary(candidate.action)
        ]
        if discretionary and not any(candidate.eligible for candidate in discretionary):
            progress = [
                candidate
                for candidate in eligible
                if _is_progress_fallback(candidate.action)
            ]
            fallback = progress or eligible
            return max(
                fallback,
                key=lambda candidate: (
                    min(
                        candidate.safety.cash_floor_margin,
                        candidate.safety.solvency_margin,
                    ),
                    candidate.value.total,
                    -candidate.semantic_priority,
                    -candidate.action,
                ),
            ).action
        return min(eligible, key=_candidate_sort_key).action

    def decide(self, env) -> Decision:
        with preserve_global_rng():
            legal = tuple(sorted(set(env.get_allowed_actions(self.player_id))))
            if not legal:
                raise RuntimeError(f"player {self.player_id} has no legal action")
            debt = getattr(env, "debt_player", None) == self.player_id
            ceiling = self._auction_ceiling(env) if env.phase == PHASE_AUCTION else None
            before_values = (
                {
                    player_id: self.value(env, player_id)
                    for player_id in range(NUM_PLAYERS)
                }
                if any(
                    action == int(ActionType.ACCEPT_TRADE) or _is_trade_offer(action)
                    for action in legal
                )
                else {}
            )
            candidates = []
            for action in legal:
                forced = (
                    debt
                    or len(legal) == 1
                    or action == int(ActionType.DECLARE_BANKRUPT)
                )
                mandatory = forced or _is_progress_fallback(action)
                if ceiling is not None:
                    candidate = self._auction_candidate(
                        env, action, forced, mandatory, ceiling
                    )
                elif action == int(ActionType.ACCEPT_TRADE) or _is_trade_offer(action):
                    candidate = self._trade_candidate(
                        env,
                        action,
                        forced,
                        mandatory,
                        before_values,
                    )
                else:
                    candidate = self._ordinary_candidate(env, action, forced, mandatory)
                candidates.append(candidate)
            frozen_candidates = tuple(candidates)
            selected = self._select(frozen_candidates, env.phase)
            rejections = tuple(
                SafetyRejection(candidate.action, candidate.rejection_reasons)
                for candidate in frozen_candidates
                if candidate.rejection_reasons
            )
            return Decision(
                policy_id=self.policy_id,
                player_id=self.player_id,
                selected_action=selected,
                candidates=frozen_candidates,
                safety_rejections=rejections,
                frozen_spec_hash=FROZEN_SPEC_HASH,
                frozen_spec_fingerprint=FROZEN_SPEC_FINGERPRINT,
            )


class ASURolloutV1:
    """Frozen strength-first rollout teacher using ASUValueV1 at every seat."""

    policy_id = ASU_ROLLOUT_V1

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.value_policy = ASUValueV1(player_id)

    def choose_action(self, env) -> int:
        return self.decide(env).selected_action

    @staticmethod
    def _shortlist(
        candidates: tuple[CandidateScore, ...],
    ) -> tuple[CandidateScore, ...]:
        ranked = sorted(
            (candidate for candidate in candidates if candidate.eligible),
            key=_candidate_sort_key,
        )
        if len(ranked) <= ROLLOUT_SHORTLIST:
            return tuple(ranked)
        mandatory = [candidate for candidate in ranked if candidate.mandatory]
        if len(mandatory) >= ROLLOUT_SHORTLIST:
            return tuple(mandatory[:ROLLOUT_SHORTLIST])
        chosen = (
            mandatory
            + [candidate for candidate in ranked if not candidate.mandatory][
                : ROLLOUT_SHORTLIST - len(mandatory)
            ]
        )
        return tuple(sorted(chosen, key=_candidate_sort_key))

    def _rollout(self, env, root_action: int, seed: int) -> float:
        game = _PrivateGame(env, seed)
        game.step(root_action)
        policies = [ASUValueV1(player_id) for player_id in range(NUM_PLAYERS)]
        for _ in range(ROLLOUT_DECISIONS):
            if game.env.done:
                break
            actor = game.env.whose_turn()
            legal = game.env.get_allowed_actions(actor)
            action = policies[actor].choose_action(game.env)
            if action not in legal:
                raise RuntimeError(
                    f"{ASU_VALUE_V1} returned illegal action {action} for seat {actor}"
                )
            game.step(action)
        return evaluate_value(game.env, self.player_id).total

    def decide(self, env) -> Decision:
        with preserve_global_rng():
            base = self.value_policy.decide(env)
            shortlist = self._shortlist(base.candidates)
            shortlist_actions = {candidate.action for candidate in shortlist}
            seeds = tuple(ROLLOUT_SEED + index for index in range(ROLLOUTS_PER_ACTION))
            updated = []
            for candidate in base.candidates:
                if candidate.action not in shortlist_actions:
                    updated.append(candidate)
                    continue
                scores = tuple(
                    self._rollout(env, candidate.action, seed) for seed in seeds
                )
                updated.append(
                    replace(
                        candidate,
                        shortlisted=True,
                        rollout_scores=scores,
                        rollout_mean=sum(scores) / len(scores),
                    )
                )
            candidates = tuple(updated)
            rolled = [candidate for candidate in candidates if candidate.shortlisted]
            selected = min(
                rolled,
                key=lambda candidate: (
                    -float(candidate.rollout_mean),
                    candidate.semantic_priority,
                    candidate.action,
                ),
            ).action
            return Decision(
                policy_id=self.policy_id,
                player_id=self.player_id,
                selected_action=selected,
                candidates=candidates,
                safety_rejections=base.safety_rejections,
                frozen_spec_hash=FROZEN_SPEC_HASH,
                frozen_spec_fingerprint=FROZEN_SPEC_FINGERPRINT,
                rollout_seeds=seeds,
            )


__all__ = [
    "ASURolloutV1",
    "ASUValueV1",
    "LONG_LAPS",
    "MINIMUM_CASH",
    "ROLLOUT_DECISIONS",
    "ROLLOUT_SEED",
    "ROLLOUT_SHORTLIST",
    "ROLLOUTS_PER_ACTION",
    "RULESET_VERSION",
    "SHORT_TURNS",
    "TERMINAL_UTILITY",
    "asset_value",
    "deed_rent",
    "evaluate_value",
    "expected_landings",
    "liquidatable_worth",
    "long_rent_projection",
    "monopoly_value",
    "movement_probabilities",
    "preserve_global_rng",
    "rent_projection",
    "safety_breakdown",
    "semantic_priority",
]
