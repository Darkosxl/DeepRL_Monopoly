from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


from monopoly_game_engine.agent_ppo import (  # noqa: E402
    PPOAgent,
    fixed_accept_trade_decision,
)
from monopoly_game_engine.actions import ActionType  # noqa: E402
from monopoly_game_engine.agents_fixed import TheGambler  # noqa: E402
from monopoly_game_engine.env import PHASE_POST_ROLL, MonopolyEnv, TradeOffer  # noqa: E402
from monopoly_game_engine.train import run_episode  # noqa: E402


class PPOAgentTests(unittest.TestCase):
    def test_hybrid_trade_policy_respects_current_legal_mask(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.turn_order = [0, 1, 2, 3]
        env.current_turn_idx = 0
        env.phase = PHASE_POST_ROLL
        env.has_rolled = False
        env.pending_trades[1] = TradeOffer(1, 0, cash_offered=100)
        agent = PPOAgent(player_id=0, hybrid=True, device="cpu")

        allowed = env.get_allowed_actions(0)
        self.assertEqual(allowed, [int(ActionType.ROLL_DICE)])
        action, log_prob, _, _ = agent.choose_action(
            env._get_state(0), env, allowed
        )

        self.assertIn(action, allowed)
        self.assertIsNotNone(log_prob)

    def test_hybrid_trade_policy_accepts_positive_recipient_value(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.pending_trades[1] = TradeOffer(1, 0, cash_offered=100)
        self.assertTrue(fixed_accept_trade_decision(env, 0))

        env.pending_trades[1] = TradeOffer(1, 0, cash_requested=100)
        self.assertFalse(fixed_accept_trade_decision(env, 0))

        gambler = TheGambler(0)
        self.assertFalse(gambler._should_accept_trade(env.pending_trades[1], env))
        self.assertTrue(
            gambler._should_accept_trade(
                TradeOffer(1, 0, cash_requested=40),
                env,
            )
        )

    def test_episode_stores_the_state_that_generated_the_action(self) -> None:
        class FakeEnv:
            max_rounds = 1

            def reset(inner_self):
                inner_self.index = 0
                inner_self.done = False
                inner_self.players = [
                    SimpleNamespace(bankrupt=False, properties=[]) for _ in range(4)
                ]
                return np.array([0.0], dtype=np.float32)

            def whose_turn(inner_self):
                return (1, 0)[inner_self.index]

            def get_allowed_actions(inner_self, _pid):
                return [int(ActionType.END_TURN)]

            def step(inner_self, _action):
                inner_self.index += 1
                inner_self.done = inner_self.index == 2
                return (
                    np.array([float(inner_self.index)], dtype=np.float32),
                    0.0,
                    inner_self.done,
                    {},
                )

            def winner(inner_self):
                return 0

            def _compute_reward(inner_self, _pid):
                return 0.0

        class FakePPO:
            player_id = 0
            n_steps = 1024

            def __init__(inner_self):
                inner_self.buffer = []
                inner_self.stored_states = []

            def choose_action(inner_self, state, _env, allowed):
                inner_self.chosen_state = state.copy()
                return allowed[0], -0.5, 0.0, allowed

            def store(inner_self, state, *_args):
                inner_self.stored_states.append(state.copy())
                inner_self.buffer.append(state)

            def update(inner_self, **_kwargs):
                return {}

            def add_win_loss(inner_self, _won):
                return None

        class FixedAgent:
            player_id = 1

            def choose_action(inner_self, _env):
                return int(ActionType.END_TURN)

        learner = FakePPO()
        run_episode(FakeEnv(), learner, [FixedAgent()], 0, is_ppo=True)

        np.testing.assert_array_equal(learner.chosen_state, np.array([1.0]))
        np.testing.assert_array_equal(learner.stored_states[0], np.array([1.0]))

    def test_cpu_update_and_checkpoint_round_trip(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        agent = PPOAgent(
            player_id=0,
            hybrid=True,
            device="cpu",
            n_epochs=1,
            batch_size=2,
        )

        for _ in range(2):
            state = env._get_state(0)
            allowed = env.get_allowed_actions(0)
            action, log_prob, value, nn_allowed = agent.choose_action(
                state, env, allowed
            )
            self.assertIn(action, allowed)
            self.assertIsNotNone(log_prob)
            agent.store(state, action, log_prob, 0.1, value, False, nn_allowed)

        stats = agent.update(last_next_state=env._get_state(0), last_done=False)
        self.assertIn("actor_loss", stats)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            agent.save(str(path))
            restored = PPOAgent(0, hybrid=True, device="cpu")
            restored.load(str(path))
            self.assertEqual(restored.step_count, agent.step_count)
            self.assertEqual(restored.games_trained, agent.games_trained)
            for expected, actual in zip(
                agent.actor.parameters(), restored.actor.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

            with self.assertRaisesRegex(ValueError, "Incompatible PPO checkpoint"):
                PPOAgent(0, hybrid=False, device="cpu").load(str(path))

    def test_legacy_checkpoint_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.pt"
            torch.save({"actor": {}, "critic": {}}, legacy)
            with self.assertRaisesRegex(ValueError, "Legacy PPO checkpoint"):
                PPOAgent(0, hybrid=True, device="cpu").load(str(legacy))


if __name__ == "__main__":
    unittest.main()
