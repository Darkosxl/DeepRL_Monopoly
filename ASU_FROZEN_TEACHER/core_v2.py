"""ASU teacher v2: faster, smarter search over the same V(s) semantics as v1.

``ASUValueV2``/``ASURolloutV2`` subclass ``ASUValueV1``/``ASURolloutV1``
(``ASU_FROZEN_TEACHER.core``) and override only the cloning and search
primitives. ``value()``, ``safety()``, ``choose_action()``,
``_ordinary_candidate()``, ``_trade_candidate()``, and the tie-break
``_select()`` are all inherited byte-identical. ``core.py`` and ``spec.py``
(including ``FROZEN_SPEC_HASH``) are never modified -- see the implementation
plan for why that has to stay true.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Mapping

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
    action_to_description,
)
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS
from monopoly_game_engine.env import PHASE_AUCTION

from .core import (
    MINIMUM_CASH,
    _candidate_sort_key,
    _dice_seeds,
    _EPSILON,
    _hypothetical_group_rent,
    _is_progress_fallback,
    _is_trade_offer,
    _safety_reasons,
    ASURolloutV1,
    ASUValueV1,
    evaluate_value,
    preserve_global_rng,
    semantic_priority,
)
from .fast_clone import _FastPrivateGame, fast_clone_env
from .spec_v2 import ASU_ROLLOUT_V2, ASU_VALUE_V2, SPEC_V2_FINGERPRINT, SPEC_V2_HASH
from .types import CandidateScore, Decision, SafetyBreakdown, SafetyRejection, ValueBreakdown


# ---------------------------------------------------------------------------
# Dice-outcome bucketing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _dice_buckets() -> tuple[tuple[tuple[int, bool], int, int], ...]:
    """Group ``core._dice_seeds()``'s 36 ordered pairs by ``(total, is_double)``.

    Each entry is ``((total, is_double), representative_seed, pair_count)``.
    Safe because ``MonopolyEnv`` only consumes randomness for ``ROLL_DICE``
    via the two ``random.randint(1, 6)`` calls in ``_do_roll``
    (monopoly_game_engine/env.py:617) -- Chance/Community Chest squares are a
    confirmed no-op in ``_handle_landing`` (env.py:683-684), and those card
    constants are otherwise unreferenced anywhere in the engine. So every
    outcome is a pure function of ``(total, is_double)``, never the raw
    ``(d1, d2)`` pair. See ``tests/test_asu_rollout_v2.py`` for the
    regression guard: the 15-bucket weighted average must equal v1's 36-pair
    average -- if the engine ever grows card-drawing logic, that test (not
    the runtime assertion in ``_roll_outcome`` below) is what should catch
    the divergence.
    """

    grouped: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for (d1, d2), seed in _dice_seeds().items():
        grouped[(d1 + d2, d1 == d2)].append(seed)
    return tuple((key, min(seeds), len(seeds)) for key, seeds in sorted(grouped.items()))


def _weighted_average_values(values: list[ValueBreakdown], weights: list[int]) -> ValueBreakdown:
    total_weight = float(sum(weights))
    fields = ("m_assets", "r_short", "r_long", "m_monopoly", "terminal_utility", "total")
    averaged = [
        sum(getattr(value, field) * weight for value, weight in zip(values, weights)) / total_weight
        for field in fields
    ]
    return ValueBreakdown(*averaged)


def _weighted_average_safety(values: list[SafetyBreakdown], weights: list[int]) -> SafetyBreakdown:
    total_weight = float(sum(weights))
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
        sum(getattr(value, field) * weight for value, weight in zip(values, weights)) / total_weight
        for field in fields
    ]
    return SafetyBreakdown(*averaged, passed=averaged[-2] >= 0 and averaged[-1] > 0)


# ---------------------------------------------------------------------------
# Trade pre-filter
# ---------------------------------------------------------------------------

TRADE_PREFILTER_SHORTLIST = 24
_TRADE_PREFILTER_SAFETY_WEIGHT = 1.0


def _group_rent_potential(
    env, party: int, color: str, overrides: Mapping[int, int | None] = {}
) -> float:
    """Cheap approximation of one color group's contribution to
    ``monopoly_value`` (core.py:424-491) for ``party``: the frozen
    ``2 ** missing`` discount applied to ``_hypothetical_group_rent`` at an
    undeveloped/unmortgaged baseline, skipping the expensive
    ``_max_developed_rent`` branch-and-bound search entirely. ``overrides``
    lets a caller ask "what if `square` were owned by `new_owner`" without
    mutating ``env``, so it can be reused for both the before and the
    after side of a hypothetical trade.
    """

    squares = tuple(COLOR_GROUPS[color])

    def owner_of(square: int) -> int | None:
        return overrides[square] if square in overrides else env.properties[square].owner

    owned = [square for square in squares if owner_of(square) == party]
    if not owned:
        return 0.0
    missing = len(squares) - len(owned)
    levels = tuple(
        env.properties[square].houses if owner_of(square) == party else 0
        for square in squares
    )
    enabled = tuple(
        owner_of(square) != party or not env.properties[square].mortgaged
        for square in squares
    )
    return _hypothetical_group_rent(color, squares, levels, enabled) / (2**missing)


def _approx_monopoly_delta(env, party: int, gained, lost) -> float:
    """Swing in ``_group_rent_potential`` for ``party`` if they gained
    ``gained`` and/or lost ``lost`` (either may be ``None``)."""

    colors = {prop.color for prop in (gained, lost) if prop is not None}
    delta = 0.0
    for color in colors:
        overrides: dict[int, int | None] = {}
        if gained is not None and gained.color == color:
            overrides[gained.square_id] = party
        if lost is not None and lost.color == color:
            overrides[lost.square_id] = None
        delta += _group_rent_potential(env, party, color, overrides) - _group_rent_potential(
            env, party, color
        )
    return delta


def _trade_prefilter_score(scratch, action: int, root: int) -> float:
    """Cheap analytic score for a not-yet-evaluated trade-offer action,
    used only to rank candidates for the shortlist -- never to decide
    eligibility or the final selected action (see ``_pruned_trade_candidate``
    and ``ASUValueV1._select``, which never lets a pruned trade win).

    ``scratch`` is a single shared clone reused across every candidate in one
    ``decide()`` call: ``_make_trade_offer``/``_make_exchange_offer``
    (monopoly_game_engine/env.py:843-896) only ever overwrite
    ``pending_trades[root]`` and read-only otherwise, so no per-candidate
    cloning or cleanup is needed between calls.
    """

    if action < OFFSETS["sell_trade"]:
        scratch._make_trade_offer(root, action - OFFSETS["buy_trade"], "buy")
    elif action < OFFSETS["exch_trade"]:
        scratch._make_trade_offer(root, action - OFFSETS["sell_trade"], "sell")
    else:
        scratch._make_exchange_offer(root, action - OFFSETS["exch_trade"])

    offer = scratch.pending_trades.get(root)
    if offer is None:
        return float("-inf")
    proposer, recipient = root, offer.to_player

    def side_score(party: int, gained, lost, cash_gained: int, cash_lost: int) -> float:
        assets_delta = (gained.price if gained is not None else 0) - (
            lost.price if lost is not None else 0
        )
        monopoly_delta = _approx_monopoly_delta(scratch, party, gained, lost)
        cash_after = scratch.players[party].cash + cash_gained - cash_lost
        safety_penalty = max(0.0, MINIMUM_CASH - cash_after) * _TRADE_PREFILTER_SAFETY_WEIGHT
        return assets_delta + monopoly_delta - safety_penalty

    proposer_score = side_score(
        proposer, offer.requested_prop, offer.offered_prop, offer.cash_requested, offer.cash_offered
    )
    recipient_score = side_score(
        recipient, offer.offered_prop, offer.requested_prop, offer.cash_offered, offer.cash_requested
    )
    # A trade that looks bad for the recipient only ever drags the proposer's
    # own score down (never up), mirroring _trade_candidate's real
    # requirement that both parties' safety gates and recipient_gain >= 0
    # eventually hold -- it never inflates a lopsided offer's rank.
    return proposer_score + min(recipient_score, 0.0)


# ---------------------------------------------------------------------------
# ASUValueV2
# ---------------------------------------------------------------------------


class ASUValueV2(ASUValueV1):
    """Faster one-step ASU teacher: identical V(s) math, cheaper search."""

    policy_id = ASU_VALUE_V2

    def _step_copy(self, env, action: int, seed: int = 0):
        game = _FastPrivateGame(env, seed)
        game.step(action)
        return game.env

    def _roll_outcome(self, env, action: int) -> tuple[ValueBreakdown, SafetyBreakdown]:
        values = []
        safety = []
        weights = []
        for (total, is_double), seed, weight in _dice_buckets():
            rolled = self._step_copy(env, action, seed)
            d1, d2 = rolled.last_dice
            if (d1 + d2, d1 == d2) != (total, is_double):
                raise AssertionError("dice seed no longer produces its frozen bucket outcome")
            values.append(self.value(rolled))
            safety.append(self.safety(rolled))
            weights.append(weight)
        return (
            _weighted_average_values(values, weights),
            _weighted_average_safety(safety, weights),
        )

    def _auction_ceiling(self, env) -> float:
        # Identical to ASUValueV1._auction_ceiling (core.py:764-776) except
        # fast_clone_env replaces copy.deepcopy.
        root = self.player_id
        square = env.auction_property_id
        if square is None:
            return 0.0
        baseline = self.value(env).total
        acquired = fast_clone_env(env)
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
        # Identical to ASUValueV1._auction_candidate (core.py:778-819) except
        # fast_clone_env replaces copy.deepcopy.
        if action == int(AuctionAction.PASS):
            after = self._step_copy(env, action)
            value = self.value(after)
            safety = self.safety(after)
            reasons: tuple[str, ...] = ()
        else:
            bid = env.auction_high_bid + AUCTION_ACTION_TO_INCREMENT[AuctionAction(action)]
            after = fast_clone_env(env)
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

    def _trade_offer_shortlist(self, env, legal: tuple[int, ...]) -> frozenset[int]:
        offers = [action for action in legal if _is_trade_offer(action)]
        if len(offers) <= TRADE_PREFILTER_SHORTLIST:
            return frozenset(offers)
        scratch = fast_clone_env(env)
        root = self.player_id
        scored = sorted(
            offers,
            key=lambda action: _trade_prefilter_score(scratch, action, root),
            reverse=True,
        )
        return frozenset(scored[:TRADE_PREFILTER_SHORTLIST])

    @staticmethod
    def _pruned_trade_candidate(
        action: int,
        mandatory: bool,
        baseline_value: ValueBreakdown,
        baseline_safety: SafetyBreakdown,
    ) -> CandidateScore:
        return CandidateScore(
            action=action,
            description=action_to_description(action),
            value=baseline_value,
            safety=baseline_safety,
            eligible=False,
            mandatory=mandatory,
            forced=False,
            semantic_priority=semantic_priority(action),
            rejection_reasons=("pruned_by_trade_prefilter",),
        )

    def decide(self, env) -> Decision:
        # Same structure as ASUValueV1.decide (core.py:899-955); the only
        # addition is routing non-shortlisted trade-offer-construction
        # actions through _pruned_trade_candidate instead of the full,
        # expensive _trade_candidate. ACCEPT_TRADE is never pruned (it's a
        # single action, not a combinatorial fan-out).
        with preserve_global_rng():
            legal = tuple(sorted(set(env.get_allowed_actions(self.player_id))))
            if not legal:
                raise RuntimeError(f"player {self.player_id} has no legal action")
            debt = getattr(env, "debt_player", None) == self.player_id
            ceiling = self._auction_ceiling(env) if env.phase == PHASE_AUCTION else None
            has_trade_actions = any(
                action == int(ActionType.ACCEPT_TRADE) or _is_trade_offer(action)
                for action in legal
            )
            before_values = (
                {player_id: self.value(env, player_id) for player_id in range(NUM_PLAYERS)}
                if has_trade_actions
                else {}
            )
            baseline_safety = self.safety(env) if has_trade_actions else None
            shortlisted = (
                self._trade_offer_shortlist(env, legal)
                if ceiling is None and has_trade_actions
                else frozenset()
            )

            candidates = []
            for action in legal:
                forced = debt or len(legal) == 1 or action == int(ActionType.DECLARE_BANKRUPT)
                mandatory = forced or _is_progress_fallback(action)
                if ceiling is not None:
                    candidate = self._auction_candidate(env, action, forced, mandatory, ceiling)
                elif action == int(ActionType.ACCEPT_TRADE):
                    candidate = self._trade_candidate(env, action, forced, mandatory, before_values)
                elif _is_trade_offer(action):
                    if forced or action in shortlisted:
                        candidate = self._trade_candidate(
                            env, action, forced, mandatory, before_values
                        )
                    else:
                        candidate = self._pruned_trade_candidate(
                            action, mandatory, before_values[self.player_id], baseline_safety
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
                frozen_spec_hash=SPEC_V2_HASH,
                frozen_spec_fingerprint=SPEC_V2_FINGERPRINT,
            )


# ---------------------------------------------------------------------------
# ASURolloutV2: successive-halving rollout allocator
# ---------------------------------------------------------------------------

ROLLOUT_SHORTLIST_V2 = 12
ROLLOUT_DECISIONS_V2 = 32
SHA_INITIAL_ROLLOUTS = 4
SHA_ROUND_MULTIPLIER = 2
SHA_ELIMINATION_Z = 1.0
SHA_MAX_TOTAL_ROLLOUTS = 128


def _mean_stderr(samples: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(samples)
    if len(samples) < 2:
        return mean, 0.0
    return mean, statistics.stdev(samples) / (len(samples) ** 0.5)


class ASURolloutV2(ASURolloutV1):
    """Successive-halving rollout teacher using ASUValueV2 at every seat.

    Unlike ASURolloutV1's fixed 8-shortlist x 8-rollouts-per-action budget,
    this races shortlisted candidates against each other: every round adds
    fresh rollouts (using the same seeds across every still-alive candidate,
    matching the "action-independent common-random-number streams"
    philosophy the frozen v1 rollout already uses -- see spec.py's
    "randomness" field) and drops candidates whose optimistic mean can no
    longer catch the leader's pessimistic mean, concentrating budget on
    genuinely close decisions instead of spending it uniformly.
    """

    policy_id = ASU_ROLLOUT_V2

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.value_policy = ASUValueV2(player_id)

    @staticmethod
    def _shortlist(candidates: tuple[CandidateScore, ...]) -> tuple[CandidateScore, ...]:
        # Same shape as ASURolloutV1._shortlist (core.py:970-989); can't be
        # inherited as-is because v1's version closes over the module-level
        # ROLLOUT_SHORTLIST constant rather than a class attribute.
        ranked = sorted(
            (candidate for candidate in candidates if candidate.eligible),
            key=_candidate_sort_key,
        )
        if len(ranked) <= ROLLOUT_SHORTLIST_V2:
            return tuple(ranked)
        mandatory = [candidate for candidate in ranked if candidate.mandatory]
        if len(mandatory) >= ROLLOUT_SHORTLIST_V2:
            return tuple(mandatory[:ROLLOUT_SHORTLIST_V2])
        chosen = (
            mandatory
            + [candidate for candidate in ranked if not candidate.mandatory][
                : ROLLOUT_SHORTLIST_V2 - len(mandatory)
            ]
        )
        return tuple(sorted(chosen, key=_candidate_sort_key))

    def _rollout(self, env, root_action: int, seed: int) -> float:
        # Identical shape to ASURolloutV1._rollout (core.py:991-1006) except
        # _FastPrivateGame replaces _PrivateGame and every seat uses
        # ASUValueV2 instead of ASUValueV1, so all of ASUValueV2's speedups
        # compound through every simulated decision.
        game = _FastPrivateGame(env, seed)
        game.step(root_action)
        policies = [ASUValueV2(player_id) for player_id in range(NUM_PLAYERS)]
        for _ in range(ROLLOUT_DECISIONS_V2):
            if game.env.done:
                break
            actor = game.env.whose_turn()
            legal = game.env.get_allowed_actions(actor)
            action = policies[actor].choose_action(game.env)
            if action not in legal:
                raise RuntimeError(
                    f"{ASU_VALUE_V2} returned illegal action {action} for seat {actor}"
                )
            game.step(action)
        return evaluate_value(game.env, self.player_id).total

    def decide(self, env) -> Decision:
        with preserve_global_rng():
            base = self.value_policy.decide(env)
            shortlist = self._shortlist(base.candidates)
            actions = [candidate.action for candidate in shortlist]
            if not actions:
                # Every candidate came back ineligible (base.decide() still
                # resolves via its own safety-fallback branch) -- fall back
                # to that one-step decision rather than crashing on an empty
                # rollout pool. ASURolloutV1 has the same latent gap; kept
                # defensive here since it's a cheap, behavior-preserving
                # guard on a path this rewrite already touches.
                return replace(
                    base,
                    policy_id=self.policy_id,
                    frozen_spec_hash=SPEC_V2_HASH,
                    frozen_spec_fingerprint=SPEC_V2_FINGERPRINT,
                )

            samples: dict[int, list[float]] = {action: [] for action in actions}
            alive = list(actions)
            round_budget = SHA_INITIAL_ROLLOUTS
            seed_cursor = 0
            total_rollouts = 0

            while alive:
                budget = round_budget
                if len(alive) * budget > SHA_MAX_TOTAL_ROLLOUTS - total_rollouts:
                    budget = max(0, (SHA_MAX_TOTAL_ROLLOUTS - total_rollouts) // len(alive))
                if budget <= 0:
                    break
                round_seeds = range(seed_cursor, seed_cursor + budget)
                for action in alive:
                    for seed in round_seeds:
                        samples[action].append(self._rollout(env, action, seed))
                seed_cursor += budget
                total_rollouts += budget * len(alive)

                if len(alive) <= 1 or total_rollouts >= SHA_MAX_TOTAL_ROLLOUTS:
                    break

                stats = {action: _mean_stderr(samples[action]) for action in alive}
                leader_action = max(alive, key=lambda action: stats[action][0])
                leader_mean, leader_stderr = stats[leader_action]
                survivors = [
                    action
                    for action in alive
                    if stats[action][0] + SHA_ELIMINATION_Z * stats[action][1]
                    >= leader_mean - SHA_ELIMINATION_Z * leader_stderr
                ]
                if len(survivors) > max(1, len(alive) // 2):
                    survivors = sorted(
                        alive, key=lambda action: stats[action][0], reverse=True
                    )[: max(1, len(alive) // 2)]
                alive = survivors
                round_budget *= SHA_ROUND_MULTIPLIER

            updated = []
            for candidate in base.candidates:
                scores = samples.get(candidate.action)
                if scores is None:
                    updated.append(candidate)
                    continue
                updated.append(
                    replace(
                        candidate,
                        shortlisted=True,
                        rollout_scores=tuple(scores),
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
                frozen_spec_hash=SPEC_V2_HASH,
                frozen_spec_fingerprint=SPEC_V2_FINGERPRINT,
                rollout_seeds=tuple(range(seed_cursor)),
            )


__all__ = [
    "ASURolloutV2",
    "ASUValueV2",
    "_dice_buckets",
]
