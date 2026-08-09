from __future__ import annotations

import hashlib
import json
import pickle
import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ASU_FROZEN_TEACHER import (
    ASURolloutV1,
    ASUValueV1,
    CandidateScore,
    FROZEN_SPEC_CANONICAL_JSON,
    FROZEN_SPEC_HASH,
    SafetyBreakdown,
    ValueBreakdown,
    asset_value,
    deed_rent,
    expected_landings,
    liquidatable_worth,
    long_rent_projection,
    monopoly_value,
)
from ASU_FROZEN_TEACHER.evaluate import (
    AgentFactory,
    AgentSpec,
    _ScriptedAdapter,
    _run_game,
    evaluate_lineup,
    parse_agent_spec,
    wilson_interval,
)
from monopoly_game_engine.actions import ActionType, AuctionAction
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_game_engine.env import (
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    MonopolyEnv,
    TradeOffer,
)
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr import (
    CFRConfig,
    MonteCarloCFR,
)


class ASUFrozenTeacherTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def give(self, square: int, player_id: int, mortgaged: bool = False) -> None:
        prop = self.env.properties[square]
        prop.owner = player_id
        prop.mortgaged = mortgaged
        self.env.players[player_id].properties.append(prop)
        self.env._update_monopolies()

    def test_frozen_hash_and_public_records_are_immutable(self) -> None:
        self.assertEqual(
            FROZEN_SPEC_HASH,
            hashlib.sha256(FROZEN_SPEC_CANONICAL_JSON.encode("ascii")).hexdigest(),
        )
        value = ValueBreakdown(1, 2, 3, 4, 0, 10)
        with self.assertRaises(FrozenInstanceError):
            value.total = 11

    def test_assets_exclude_cash_and_mortgaged_holdings(self) -> None:
        self.give(1, 0)
        self.env.properties[1].houses = 2
        self.assertEqual(asset_value(self.env, 0), 160)
        self.env.properties[1].mortgaged = True
        self.assertEqual(asset_value(self.env, 0), 0)

    def test_complete_turn_projection_handles_doubles_and_jail(self) -> None:
        player = self.env.players[0]
        player.position = 0
        one_turn = expected_landings(player, 1)
        self.assertAlmostEqual(sum(one_turn.values()), 257 / 216)

        player.position = 10
        player.in_jail = True
        player.jail_turns = 0
        self.assertAlmostEqual(sum(expected_landings(player, 1).values()), 1 / 6)
        self.assertAlmostEqual(
            sum(expected_landings(player, 5).values()),
            3.8101810214835843,
        )

    def test_railroad_and_utility_rent_use_counts_and_dice(self) -> None:
        for square in (5, 15):
            self.give(square, 0)
        self.assertEqual(deed_rent(self.env, 5, 7), 50)
        for square in (12, 28):
            self.give(square, 1)
        self.assertEqual(deed_rent(self.env, 12, 8), 80)
        self.env.properties[12].mortgaged = True
        self.assertEqual(deed_rent(self.env, 12, 8), 0)

    def test_five_lap_projection_uses_fifty_over_forty_nine_visits(self) -> None:
        self.give(5, 0)
        projection = long_rent_projection(self.env, 0)
        self.assertAlmostEqual(projection.income, 3 * 25 * 50 / 49)
        self.assertEqual(projection.exposure, 0)

    def test_monopoly_discount_is_two_to_the_missing_count(self) -> None:
        self.env.houses_available = 0
        self.env.hotels_available = 0
        self.give(1, 0)
        self.assertEqual(monopoly_value(self.env, 0, r_long=0), 6)
        self.give(3, 0)
        self.assertEqual(monopoly_value(self.env, 0, r_long=0), 12)

    def test_value_reports_all_components_and_terminal_utility(self) -> None:
        self.give(1, 0)
        value = ASUValueV1(0).value(self.env)
        self.assertEqual(
            value.total,
            value.m_assets + value.r_short + value.r_long + value.m_monopoly,
        )
        self.env.done = True
        self.env.players[1].bankrupt = True
        self.env.players[2].bankrupt = True
        self.env.players[3].bankrupt = True
        self.assertEqual(ASUValueV1(0).value(self.env).terminal_utility, 1_000_000)

    def test_liquidation_includes_house_sales_and_mortgage_value(self) -> None:
        self.give(1, 0)
        self.env.properties[1].houses = 2
        self.assertEqual(liquidatable_worth(self.env, 0), 80)
        self.env.properties[1].mortgaged = True
        self.env.properties[1].houses = 0
        self.assertEqual(liquidatable_worth(self.env, 0), 0)

    def test_bankruptcy_gate_vetoes_unsafe_purchase(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = True
        self.env.players[0].position = 1
        self.env.players[0].cash = 250
        decision = ASUValueV1(0).decide(self.env)
        purchase = next(
            candidate
            for candidate in decision.candidates
            if candidate.action == int(ActionType.BUY_PROPERTY)
        )
        self.assertFalse(purchase.eligible)
        self.assertEqual(decision.action, int(ActionType.END_TURN))

    def test_debt_liquidation_is_never_pruned(self) -> None:
        self.give(1, 0)
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = True
        self.env.debt_player = 0
        self.env.debt_creditor = 1
        self.env.debt_amount = 30
        self.env.players[0].cash = 0
        decision = ASUValueV1(0).decide(self.env)
        self.assertTrue(decision.candidates)
        self.assertTrue(
            all(item.forced and item.eligible for item in decision.candidates)
        )

    def test_mutually_beneficial_trade_is_accepted(self) -> None:
        self.env.turn_order = [1, 0, 2, 3]
        self.env.current_turn_idx = 0
        self.env.phase = PHASE_OUT_OF_TURN
        self.env.out_of_turn_pids = [0]
        for square in (1, 21, 23):
            self.give(square, 1)
        self.env.players[1].cash = 150
        self.env.pending_trades[1] = TradeOffer(
            1,
            0,
            offered_prop=self.env.properties[1],
            cash_requested=75,
        )
        decision = ASUValueV1(0).decide(self.env)
        accepted = next(
            item
            for item in decision.candidates
            if item.action == int(ActionType.ACCEPT_TRADE)
        )
        self.assertGreater(accepted.proposer_gain, 0)
        self.assertGreaterEqual(accepted.recipient_gain, 0)
        self.assertTrue(accepted.eligible)
        self.assertEqual(decision.action, int(ActionType.ACCEPT_TRADE))

    def test_rejected_trade_uses_semantic_safe_fallback(self) -> None:
        self.env.turn_order = [1, 0, 2, 3]
        self.env.current_turn_idx = 0
        self.env.phase = PHASE_OUT_OF_TURN
        self.env.out_of_turn_pids = [0]
        self.env.pending_trades[1] = TradeOffer(1, 0, cash_requested=100)
        decision = ASUValueV1(0).decide(self.env)
        self.assertEqual(decision.action, int(ActionType.DECLINE_TRADE))

    def test_auction_uses_largest_safe_increment_under_ceiling(self) -> None:
        self.env._start_auction(1)
        self.env.players[0].cash = 250
        decision = ASUValueV1(0).decide(self.env)
        self.assertEqual(decision.action, int(AuctionAction.BID_50))
        bid_100 = next(
            item
            for item in decision.candidates
            if item.action == int(AuctionAction.BID_100)
        )
        self.assertFalse(bid_100.eligible)

    def test_value_decisions_preserve_environment_and_global_rngs(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False
        source = pickle.dumps(self.env, protocol=5)
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state(), protocol=5)
        torch_state = torch.get_rng_state().clone()

        first = ASUValueV1(0).decide(self.env)
        second = ASUValueV1(0).decide(self.env)

        self.assertEqual(first, second)
        self.assertEqual(source, pickle.dumps(self.env, protocol=5))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(numpy_state, pickle.dumps(np.random.get_state(), protocol=5))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_full_budget_rollout_is_repeatable_and_uses_common_streams(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = True
        self.env.players[0].position = 1
        source = pickle.dumps(self.env, protocol=5)
        state = random.getstate()
        first = ASURolloutV1(0).decide(self.env)
        second = ASURolloutV1(0).decide(self.env)
        self.assertEqual(first, second)
        self.assertEqual(first.rollout_seeds, tuple(range(8)))
        shortlisted = [item for item in first.candidates if item.shortlisted]
        self.assertGreaterEqual(len(shortlisted), 2)
        self.assertTrue(all(len(item.rollout_scores) == 8 for item in shortlisted))
        self.assertEqual(source, pickle.dumps(self.env, protocol=5))
        self.assertEqual(state, random.getstate())

    def test_exact_value_tie_uses_lower_action_id(self) -> None:
        value = ValueBreakdown(1, 2, 3, 4, 0, 10)
        safety = SafetyBreakdown(500, 0, 0, 0, 0, 300, 500, True)
        low, high = sorted((int(ActionType.DO_NOTHING), int(ActionType.END_TURN)))
        candidates = tuple(
            CandidateScore(
                action=action,
                description=str(action),
                value=value,
                safety=safety,
                eligible=True,
                mandatory=False,
                forced=False,
                semantic_priority=50,
            )
            for action in (high, low)
        )
        self.assertEqual(ASUValueV1._select(candidates, PHASE_POST_ROLL), low)

    def test_one_full_value_game_completes_without_truncation(self) -> None:
        random.seed(23)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        agents = [ASUValueV1(0), FPAgentA(1), FPAgentB(2), FPAgentC(3)]
        decisions = 0
        while not env.done and decisions < 20_000:
            player_id = env.whose_turn()
            allowed = env.get_allowed_actions(player_id)
            action = agents[player_id].choose_action(env)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else min(allowed)
                )
            env.step(action)
            decisions += 1
        self.assertTrue(env.done)
        self.assertLess(decisions, 20_000)


class ASUEvaluatorTests(unittest.TestCase):
    def test_seat_rotation_json_and_wilson_interval(self) -> None:
        result = evaluate_lineup(
            "fixed-a",
            ("fixed-b", "fixed-c", "fixed-d"),
            seeds=(3,),
            max_decisions=1,
        )
        self.assertEqual(result["ruleset"], "ppo-plus-v2")
        self.assertEqual(result["frozen_spec_hash"], FROZEN_SPEC_HASH)
        self.assertEqual([game["focus_seat"] for game in result["games"]], [0, 1, 2, 3])
        self.assertEqual(result["truncations"], 4)
        self.assertEqual(
            result["scripted_compatibility_fallbacks"],
            sum(
                sum(game["scripted_compatibility_fallbacks"])
                for game in result["games"]
            ),
        )
        self.assertTrue(all(game["winner"] is None for game in result["games"]))
        self.assertTrue(all(game["focus_won"] is None for game in result["games"]))
        self.assertTrue(
            all(game["provisional_leader"] is not None for game in result["games"])
        )
        focus_summary = result["win_rates"]["fixed-a"]
        self.assertEqual(focus_summary["games"], 0)
        self.assertEqual(focus_summary["truncated_appearances"], 4)
        self.assertIsNone(focus_summary["win_rate"])
        json.dumps(result)
        low, high = wilson_interval(1, 4)
        self.assertLess(low, 0.25)
        self.assertGreater(high, 0.25)

    def test_missing_checkpoint_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing PPO checkpoint"):
            parse_agent_spec("ppo:/definitely/missing/checkpoint.pt")

    def test_evaluator_requires_exactly_three_opponents(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 3 opponents"):
            evaluate_lineup(
                "fixed-a",
                ("fixed-b", "fixed-c"),  # type: ignore[arg-type]
                max_decisions=1,
            )

    def test_illegal_policy_output_is_rejected(self) -> None:
        class IllegalFactory:
            @staticmethod
            def build(_spec, _seat):
                return SimpleNamespace(choose_action=lambda _env: 999_999)

        specs = tuple(AgentSpec("fixed-a", "fixed-a") for _ in range(4))
        with self.assertRaisesRegex(RuntimeError, "returned illegal action"):
            _run_game(specs, 0, 3, 1, IllegalFactory())  # type: ignore[arg-type]

    def test_scripted_adapter_uses_and_counts_historical_fallback(self) -> None:
        class ForcedEnv:
            def __init__(self, allowed):
                self.allowed = allowed

            def get_allowed_actions(self, _player_id):
                return self.allowed

        raw = SimpleNamespace(choose_action=lambda _env: int(ActionType.ROLL_DICE))
        adapter = _ScriptedAdapter(raw, player_id=2)
        self.assertEqual(
            adapter.choose_action(ForcedEnv([int(ActionType.ROLL_DICE)])),
            int(ActionType.ROLL_DICE),
        )
        self.assertEqual(adapter.fallbacks, 0)

        self.assertEqual(
            adapter.choose_action(
                ForcedEnv([int(ActionType.END_TURN), int(ActionType.DECLARE_BANKRUPT)])
            ),
            int(ActionType.END_TURN),
        )
        self.assertEqual(
            adapter.choose_action(ForcedEnv([int(ActionType.DECLARE_BANKRUPT)])),
            int(ActionType.DECLARE_BANKRUPT),
        )
        self.assertEqual(adapter.fallbacks, 2)

    def test_ddqn_checkpoint_adapter_can_bind_a_different_seat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ddqn.pt"
            DDQNAgent(0, hidden_dim=16, device="cpu").save(str(checkpoint))
            adapter = AgentFactory().build(
                parse_agent_spec(f"ddqn:{checkpoint}"), player_id=2
            )
            env = MonopolyEnv(agent_ids=[2], max_rounds=1)
            env.turn_order = [2, 0, 1, 3]
            env.current_turn_idx = 0
            action = adapter.choose_action(env)
            self.assertIn(action, env.get_allowed_actions(2))

    def test_ppo_and_cfr_checkpoint_adapters_bind_a_different_seat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ppo_checkpoint = root / "ppo.pt"
            cfr_checkpoint = root / "cfr.pkl.gz"
            PPOAgent(0, hidden_dim=16, device="cpu").save(str(ppo_checkpoint))
            MonteCarloCFR(CFRConfig(max_rounds=1, max_decisions=1)).save(cfr_checkpoint)

            env = MonopolyEnv(agent_ids=[2], max_rounds=1)
            env.turn_order = [2, 0, 1, 3]
            env.current_turn_idx = 0
            for kind, checkpoint in (
                ("ppo", ppo_checkpoint),
                ("cfr", cfr_checkpoint),
            ):
                with self.subTest(kind=kind):
                    adapter = AgentFactory().build(
                        parse_agent_spec(f"{kind}:{checkpoint}"), player_id=2
                    )
                    action = adapter.choose_action(env)
                    self.assertIn(action, env.get_allowed_actions(2))


if __name__ == "__main__":
    unittest.main()
