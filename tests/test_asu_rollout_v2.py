from __future__ import annotations

import copy
import pickle
import random
import unittest

import numpy as np
import torch

from ASU_FROZEN_TEACHER import ASUValueV1
from ASU_FROZEN_TEACHER.core import _dice_seeds, _is_trade_offer
from ASU_FROZEN_TEACHER.core_v2 import (
    SHA_MAX_TOTAL_ROLLOUTS,
    TRADE_PREFILTER_SHORTLIST,
    ASURolloutV2,
    ASUValueV2,
    _dice_buckets,
    _trade_prefilter_score,
)
from ASU_FROZEN_TEACHER.fast_clone import fast_clone_env
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTY_IDS
from monopoly_game_engine.env import PHASE_POST_ROLL, MonopolyEnv, TradeOffer


def _snapshot(env: MonopolyEnv) -> dict:
    """Extract every comparable field of ``env`` by value, not identity."""

    return {
        "agent_ids": list(env.agent_ids),
        "max_rounds": env.max_rounds,
        "current_turn_idx": env.current_turn_idx,
        "round": env.round,
        "done": env.done,
        "last_dice": env.last_dice,
        "phase": env.phase,
        "has_rolled": env.has_rolled,
        "turn_order": list(env.turn_order),
        "out_of_turn_pids": list(env.out_of_turn_pids),
        "consecutive_doubles": env.consecutive_doubles,
        "extra_roll_pending": env.extra_roll_pending,
        "auction_property_id": env.auction_property_id,
        "auction_bidders": list(env.auction_bidders),
        "auction_current_pid": env.auction_current_pid,
        "auction_high_bid": env.auction_high_bid,
        "auction_high_bidder": env.auction_high_bidder,
        "houses_available": env.houses_available,
        "hotels_available": env.hotels_available,
        "debt_player": env.debt_player,
        "debt_creditor": env.debt_creditor,
        "debt_amount": env.debt_amount,
        "player_needs_funds": env.player_needs_funds,
        "properties": {
            square: (
                prop.square_id,
                prop.name,
                prop.price,
                prop.mortgage_v,
                prop.color,
                prop.owner,
                prop.mortgaged,
                prop.houses,
                prop.is_monopoly,
            )
            for square, prop in env.properties.items()
        },
        "players": [
            (
                player.player_id,
                player.cash,
                player.position,
                player.in_jail,
                player.jail_turns,
                player.gooj_card,
                player.bankrupt,
                tuple(prop.square_id for prop in player.properties),
            )
            for player in env.players
        ],
        "pending_trades": {
            sender: (
                offer.from_player,
                offer.to_player,
                offer.offered_prop.square_id if offer.offered_prop else None,
                offer.requested_prop.square_id if offer.requested_prop else None,
                offer.cash_offered,
                offer.cash_requested,
            )
            for sender, offer in env.pending_trades.items()
        },
    }


def _new_env() -> MonopolyEnv:
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    return env


def _give(env: MonopolyEnv, square: int, player_id: int, *, houses: int = 0, mortgaged: bool = False) -> None:
    prop = env.properties[square]
    prop.owner = player_id
    prop.houses = houses
    prop.mortgaged = mortgaged
    env.players[player_id].properties.append(prop)
    env._update_monopolies()


def _rich_states() -> list[MonopolyEnv]:
    """A handful of hand-built states exercising every kind of mutable field."""

    states = []

    fresh = _new_env()
    states.append(fresh)

    developed = _new_env()
    _give(developed, 1, 0, houses=2)
    _give(developed, 3, 0)
    _give(developed, 5, 1, mortgaged=True)
    developed.players[0].cash = 850
    states.append(developed)

    with_trade = _new_env()
    _give(with_trade, 6, 1)
    _give(with_trade, 8, 0)
    with_trade.pending_trades[1] = TradeOffer(
        1, 0, offered_prop=with_trade.properties[6], cash_requested=50
    )
    with_trade.pending_trades[2] = TradeOffer(
        2, 3, requested_prop=with_trade.properties[8], cash_offered=25
    )
    states.append(with_trade)

    auctioning = _new_env()
    auctioning._start_auction(11)
    auctioning.auction_high_bid = 40
    auctioning.auction_high_bidder = 2
    states.append(auctioning)

    indebted = _new_env()
    _give(indebted, 9, 1)
    indebted.debt_player = 0
    indebted.debt_creditor = 1
    indebted.debt_amount = 60
    indebted.player_needs_funds = 0
    states.append(indebted)

    bankrupt = _new_env()
    _give(bankrupt, 13, 2)
    _give(bankrupt, 14, 2)
    bankrupt.players[0].bankrupt = True
    bankrupt.players[0].cash = 0
    states.append(bankrupt)

    return states


class FastCloneTests(unittest.TestCase):
    def test_fast_clone_matches_deepcopy_snapshot(self) -> None:
        for index, env in enumerate(_rich_states()):
            with self.subTest(state=index):
                self.assertEqual(_snapshot(copy.deepcopy(env)), _snapshot(fast_clone_env(env)))

    def test_fast_clone_preserves_object_identity_invariants(self) -> None:
        env = _new_env()
        _give(env, 6, 1)
        _give(env, 8, 0)
        env.pending_trades[1] = TradeOffer(
            1, 0, offered_prop=env.properties[6], requested_prop=env.properties[8], cash_requested=50
        )

        clone = fast_clone_env(env)

        prop6 = clone.properties[6]
        prop8 = clone.properties[8]
        self.assertIs(clone.players[1].properties[0], prop6)
        self.assertIs(clone.players[0].properties[0], prop8)
        offer = clone.pending_trades[1]
        self.assertIs(offer.offered_prop, prop6)
        self.assertIs(offer.requested_prop, prop8)
        # And the clone must not alias the original's objects.
        self.assertIsNot(prop6, env.properties[6])
        self.assertIsNot(clone.players[1], env.players[1])

    def test_fast_clone_is_fully_isolated_from_original(self) -> None:
        env = _new_env()
        _give(env, 1, 0)
        clone = fast_clone_env(env)

        clone.properties[1].houses = 3
        clone.players[0].cash = 1
        clone.players[0].properties.append(clone.properties[3])
        clone.turn_order.append(99)
        self.assertEqual(env.properties[1].houses, 0)
        self.assertEqual(env.players[0].cash, 1500)
        self.assertEqual(len(env.players[0].properties), 1)
        self.assertNotIn(99, env.turn_order)

        env.properties[1].mortgaged = True
        env.players[0].cash = 2
        env.turn_order.append(77)
        self.assertFalse(clone.properties[1].mortgaged)
        self.assertEqual(clone.players[0].cash, 1)
        self.assertNotIn(77, clone.turn_order)

    def test_fast_clone_step_matches_deepcopy_step_under_identical_rng(self) -> None:
        for seed in range(60):
            random.seed(seed)
            env = _new_env()
            for _ in range(random.randint(0, 40)):
                if env.done:
                    break
                actor = env.whose_turn()
                allowed = env.get_allowed_actions(actor)
                if not allowed:
                    break
                env.step(random.choice(allowed))
            if env.done:
                continue
            actor = env.whose_turn()
            allowed = env.get_allowed_actions(actor)
            if not allowed:
                continue
            action = random.choice(allowed)

            deep = copy.deepcopy(env)
            fast = fast_clone_env(env)

            step_seed = random.Random(f"step-{seed}").getstate()
            random.setstate(step_seed)
            deep.step(action)
            state_after_deep = random.getstate()

            random.setstate(step_seed)
            fast.step(action)
            state_after_fast = random.getstate()

            with self.subTest(seed=seed):
                self.assertEqual(_snapshot(deep), _snapshot(fast))
                self.assertEqual(state_after_deep, state_after_fast)


class DiceBucketTests(unittest.TestCase):
    def test_buckets_cover_all_36_pairs_grouped_by_total_and_doubles(self) -> None:
        buckets = _dice_buckets()
        self.assertEqual(len(buckets), 15)
        self.assertEqual(sum(weight for _, _, weight in buckets), 36)
        expected: dict[tuple[int, bool], int] = {}
        for d1, d2 in _dice_seeds():
            key = (d1 + d2, d1 == d2)
            expected[key] = expected.get(key, 0) + 1
        actual = {key: weight for key, _, weight in buckets}
        self.assertEqual(actual, expected)

    def test_weighted_dice_average_matches_unweighted_36_pair_average(self) -> None:
        # Skip states where player 0 (the acting/evaluated seat) is bankrupt:
        # a bankrupt player is never actually whose_turn() in reachable play
        # (they're skipped by the engine's own turn-advancement), so forcing
        # ROLL_DICE for them here would exercise an engine-inconsistent state
        # that even ASUValueV1's own frozen dice-seed assertion (core.py:696-
        # 697) correctly rejects -- unrelated to the bucketing being tested.
        for index, env in enumerate(_rich_states()):
            if env.players[0].bankrupt:
                continue
            env.phase = PHASE_POST_ROLL
            env.has_rolled = False
            with self.subTest(state=index):
                v1_value, v1_safety = ASUValueV1(0)._roll_outcome(env, int(ActionType.ROLL_DICE))
                v2_value, v2_safety = ASUValueV2(0)._roll_outcome(env, int(ActionType.ROLL_DICE))
                for field in ("m_assets", "r_short", "r_long", "m_monopoly", "terminal_utility", "total"):
                    self.assertAlmostEqual(
                        getattr(v1_value, field), getattr(v2_value, field), places=9
                    )
                for field in (
                    "cash_after",
                    "next_round_net_rent",
                    "next_round_rent_income",
                    "liquidatable_worth",
                    "worst_reachable_rent",
                    "cash_floor_margin",
                    "solvency_margin",
                ):
                    self.assertAlmostEqual(
                        getattr(v1_safety, field), getattr(v2_safety, field), places=9
                    )
                self.assertEqual(v1_safety.passed, v2_safety.passed)


_VALUE_FIELDS = ("m_assets", "r_short", "r_long", "m_monopoly", "terminal_utility", "total")
_SAFETY_FIELDS = (
    "cash_after",
    "next_round_net_rent",
    "next_round_rent_income",
    "liquidatable_worth",
    "worst_reachable_rent",
    "cash_floor_margin",
    "solvency_margin",
)
_CANDIDATE_EXACT_FIELDS = (
    "action",
    "description",
    "eligible",
    "mandatory",
    "forced",
    "semantic_priority",
    "rejection_reasons",
    "proposer_gain",
    "recipient_gain",
    "auction_ceiling",
    "shortlisted",
)


class ASUValueV2EquivalenceTests(unittest.TestCase):
    """Before the trade pre-filter is layered in, v2 must decide identically
    to v1 -- the only differences so far (fast clone, bucketed dice) are
    meant to be pure speedups, not exact-bitwise-identical behavior. Dice
    outcomes specifically are compared with float tolerance: v1 averages 36
    equally-weighted pairs and v2 averages 15 differently-weighted buckets,
    which are mathematically but not bit-for-bit identical (summation
    order/grouping differs)."""

    def _assert_equivalent_decision(self, env: MonopolyEnv, player_id: int = 0) -> None:
        v1 = ASUValueV1(player_id).decide(env)
        v2 = ASUValueV2(player_id).decide(env)
        self.assertEqual(v1.selected_action, v2.selected_action)
        self.assertEqual(len(v1.candidates), len(v2.candidates))
        for one, two in zip(v1.candidates, v2.candidates):
            for field in _CANDIDATE_EXACT_FIELDS:
                self.assertEqual(
                    getattr(one, field), getattr(two, field), msg=f"field {field} on action {one.action}"
                )
            for field in _VALUE_FIELDS:
                self.assertAlmostEqual(
                    getattr(one.value, field),
                    getattr(two.value, field),
                    places=6,
                    msg=f"value.{field} on action {one.action}",
                )
            for field in _SAFETY_FIELDS:
                self.assertAlmostEqual(
                    getattr(one.safety, field),
                    getattr(two.safety, field),
                    places=6,
                    msg=f"safety.{field} on action {one.action}",
                )
            self.assertEqual(one.safety.passed, two.safety.passed)
        self.assertEqual(
            [item.action for item in v1.safety_rejections],
            [item.action for item in v2.safety_rejections],
        )

    def test_matches_v1_on_rich_states(self) -> None:
        for index, env in enumerate(_rich_states()):
            with self.subTest(state=index):
                self._assert_equivalent_decision(env)

    def test_matches_v1_with_roll_dice_legal(self) -> None:
        env = _new_env()
        _give(env, 1, 0)
        env.phase = PHASE_POST_ROLL
        env.has_rolled = False
        self._assert_equivalent_decision(env)

    def test_matches_v1_during_auction(self) -> None:
        env = _new_env()
        env._start_auction(3)
        env.players[0].cash = 300
        self._assert_equivalent_decision(env)

    def test_matches_v1_across_random_playouts(self) -> None:
        for seed in range(20):
            random.seed(seed)
            env = _new_env()
            for _ in range(random.randint(0, 30)):
                if env.done:
                    break
                actor = env.whose_turn()
                allowed = env.get_allowed_actions(actor)
                if not allowed:
                    break
                env.step(random.choice(allowed))
            if env.done or not env.get_allowed_actions(env.whose_turn()):
                continue
            with self.subTest(seed=seed):
                self._assert_equivalent_decision(env, env.whose_turn())


def _real_estate_group(size: int) -> tuple[str, tuple[int, ...]]:
    for color, squares in COLOR_GROUPS.items():
        if color not in ("railroad", "utility") and len(squares) == size:
            return color, tuple(squares)
    raise AssertionError(f"no real-estate color group of size {size}")


def _decode_trade_offer(env: MonopolyEnv, action: int, pid: int) -> TradeOffer:
    probe = fast_clone_env(env)
    if action < OFFSETS["sell_trade"]:
        probe._make_trade_offer(pid, action - OFFSETS["buy_trade"], "buy")
    elif action < OFFSETS["exch_trade"]:
        probe._make_trade_offer(pid, action - OFFSETS["sell_trade"], "sell")
    else:
        probe._make_exchange_offer(pid, action - OFFSETS["exch_trade"])
    return probe.pending_trades[pid]


class TradePrefilterTests(unittest.TestCase):
    def _large_fan_out_env(self) -> MonopolyEnv:
        env = _new_env()
        for square in PROPERTY_IDS[0:3]:
            _give(env, square, 0)
        for square in PROPERTY_IDS[3:9]:
            _give(env, square, 1)
        for player in env.players:
            player.cash = 1500
        return env

    def test_large_fan_out_exceeds_shortlist_size(self) -> None:
        env = self._large_fan_out_env()
        legal = tuple(sorted(set(env.get_allowed_actions(0))))
        offers = [action for action in legal if _is_trade_offer(action)]
        self.assertGreater(len(offers), TRADE_PREFILTER_SHORTLIST)

    def test_shortlist_is_capped_and_subset_of_legal_offers(self) -> None:
        env = self._large_fan_out_env()
        legal = tuple(sorted(set(env.get_allowed_actions(0))))
        offers = {action for action in legal if _is_trade_offer(action)}
        shortlist = ASUValueV2(0)._trade_offer_shortlist(env, legal)
        self.assertEqual(len(shortlist), TRADE_PREFILTER_SHORTLIST)
        self.assertTrue(shortlist.issubset(offers))

    def test_small_fan_out_is_not_pruned_at_all(self) -> None:
        env = _new_env()
        _give(env, 1, 0)
        _give(env, 3, 1)
        for player in env.players:
            player.cash = 1500
        legal = tuple(sorted(set(env.get_allowed_actions(0))))
        offers = {action for action in legal if _is_trade_offer(action)}
        self.assertLessEqual(len(offers), TRADE_PREFILTER_SHORTLIST)
        shortlist = ASUValueV2(0)._trade_offer_shortlist(env, legal)
        self.assertEqual(shortlist, offers)

    def test_pruned_candidates_are_marked_ineligible_with_reason(self) -> None:
        env = self._large_fan_out_env()
        decision = ASUValueV2(0).decide(env)
        pruned = [
            candidate
            for candidate in decision.candidates
            if candidate.rejection_reasons == ("pruned_by_trade_prefilter",)
        ]
        self.assertTrue(pruned)
        self.assertTrue(all(not candidate.eligible for candidate in pruned))
        self.assertTrue(all(_is_trade_offer(candidate.action) for candidate in pruned))

    def test_decide_never_selects_a_pruned_trade(self) -> None:
        for seed in range(10):
            random.seed(seed)
            env = self._large_fan_out_env()
            decision = ASUValueV2(0).decide(env)
            selected = decision.selected
            with self.subTest(seed=seed):
                self.assertNotEqual(selected.rejection_reasons, ("pruned_by_trade_prefilter",))

    def test_shortlist_recall_prioritizes_the_monopoly_completing_trade(self) -> None:
        _, squares = _real_estate_group(3)
        env = _new_env()
        _give(env, squares[0], 0)
        _give(env, squares[1], 0)
        _give(env, squares[2], 1)
        filler = [square for square in PROPERTY_IDS if square not in squares][:8]
        for square in filler:
            _give(env, square, 1)
        for player in env.players:
            player.cash = 3000

        legal = tuple(sorted(set(env.get_allowed_actions(0))))
        offers = [action for action in legal if _is_trade_offer(action)]
        self.assertGreater(len(offers), TRADE_PREFILTER_SHORTLIST)

        scratch = fast_clone_env(env)
        scored = sorted(
            offers, key=lambda action: _trade_prefilter_score(scratch, action, 0), reverse=True
        )
        top_action = scored[0]
        top_offer = _decode_trade_offer(env, top_action, 0)
        self.assertIsNotNone(top_offer.requested_prop)
        self.assertEqual(top_offer.requested_prop.square_id, squares[2])

        shortlist = ASUValueV2(0)._trade_offer_shortlist(env, legal)
        self.assertIn(top_action, shortlist)


class ASURolloutV2Tests(unittest.TestCase):
    def _rollout_ready_env(self) -> MonopolyEnv:
        env = _new_env()
        env.phase = PHASE_POST_ROLL
        env.has_rolled = True
        env.players[0].position = 1
        return env

    def test_decide_is_repeatable_and_preserves_rngs(self) -> None:
        env = self._rollout_ready_env()
        source = pickle.dumps(env, protocol=5)
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state(), protocol=5)
        torch_state = torch.get_rng_state().clone()

        first = ASURolloutV2(0).decide(env)
        second = ASURolloutV2(0).decide(env)

        self.assertEqual(first, second)
        self.assertEqual(source, pickle.dumps(env, protocol=5))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(numpy_state, pickle.dumps(np.random.get_state(), protocol=5))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_selected_action_is_legal_and_shortlisted(self) -> None:
        env = self._rollout_ready_env()
        decision = ASURolloutV2(0).decide(env)
        self.assertIn(decision.selected_action, env.get_allowed_actions(0))
        self.assertTrue(decision.selected.shortlisted)
        self.assertIsNotNone(decision.selected.rollout_mean)

    def test_total_rollouts_never_exceeds_hard_cap(self) -> None:
        for seed in range(5):
            random.seed(seed)
            env = self._rollout_ready_env()
            decision = ASURolloutV2(0).decide(env)
            total = sum(
                len(candidate.rollout_scores)
                for candidate in decision.candidates
                if candidate.shortlisted
            )
            with self.subTest(seed=seed):
                self.assertLessEqual(total, SHA_MAX_TOTAL_ROLLOUTS)

    def test_every_shortlisted_candidate_has_at_least_one_sample(self) -> None:
        env = self._rollout_ready_env()
        decision = ASURolloutV2(0).decide(env)
        shortlisted = [candidate for candidate in decision.candidates if candidate.shortlisted]
        self.assertGreaterEqual(len(shortlisted), 1)
        self.assertTrue(all(len(candidate.rollout_scores) >= 1 for candidate in shortlisted))

    def test_bounded_playout_has_no_illegal_actions(self) -> None:
        # Not a full game: ASURolloutV2 costs real wall-clock time per
        # decision even after the v1->v2 speedup, and the original ASU
        # test suite makes the same call for the same reason (its full-game
        # smoke test uses ASUValueV1, not ASURolloutV1 --
        # tests/test_asu_frozen_teacher.py:272-290). This exercises enough
        # real turns (opponent moves, property purchases, rent, several of
        # player 0's own rollout decisions) to catch an illegal-action bug
        # without paying for an entire game.
        random.seed(11)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        agents = [ASURolloutV2(0), ASUValueV2(1), ASUValueV2(2), ASUValueV2(3)]
        for _ in range(24):
            if env.done:
                break
            actor = env.whose_turn()
            allowed = env.get_allowed_actions(actor)
            action = agents[actor].choose_action(env)
            self.assertIn(action, allowed)
            env.step(action)


if __name__ == "__main__":
    unittest.main()
