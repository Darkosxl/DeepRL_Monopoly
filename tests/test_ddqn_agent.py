from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch


from monopoly_game_engine.actions import ACTION_SPACE_SIZE, OFFSETS, ActionType  # noqa: E402
from monopoly_game_engine.agent_ddqn import DDQNAgent  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv, TradeOffer  # noqa: E402
from monopoly_game_engine.networks import DDQNNetwork  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402
from monopoly_game_engine.train import run_episode  # noqa: E402


class DDQNAgentTests(unittest.TestCase):
    def test_paper_network_shape(self) -> None:
        network = DDQNNetwork()
        self.assertEqual(network.net[0].in_features, STATE_DIM)
        self.assertEqual(network.net[0].out_features, 1024)
        self.assertEqual(network.net[2].out_features, 512)
        self.assertEqual(network.net[4].out_features, ACTION_SPACE_SIZE)

    def test_next_action_argmax_respects_legal_mask(self) -> None:
        q_values = torch.zeros(2, ACTION_SPACE_SIZE)
        q_values[0, 99] = 100
        q_values[0, 7] = 5
        q_values[1, 99] = 100

        actions = DDQNAgent._masked_argmax(q_values, ((7,), ()))

        self.assertEqual(actions.tolist(), [7, int(ActionType.DO_NOTHING)])

    def test_exploration_balances_action_sections(self) -> None:
        allowed = [
            int(ActionType.END_TURN),
            OFFSETS["mortgage"],
            OFFSETS["mortgage"] + 1,
            OFFSETS["exch_trade"],
            OFFSETS["exch_trade"] + 1,
        ]
        choices = []

        def choose_first(options):
            choices.append(list(options))
            return options[0]

        with patch(
            "monopoly_game_engine.agent_ddqn.random.choice", side_effect=choose_first
        ):
            action = DDQNAgent._balanced_random_action(allowed)

        self.assertEqual(action, int(ActionType.END_TURN))
        self.assertEqual(len(choices[0]), 3)
        self.assertEqual(len(choices[1]), 1)

    def test_hybrid_trade_only_intercepts_when_legal(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.pending_trades[1] = TradeOffer(1, 0, cash_offered=100)
        agent = DDQNAgent(0, hybrid=True, hidden_dim=16, device="cpu")
        allowed = [int(ActionType.ROLL_DICE)]

        action, neural_mask = agent.choose_training_action(
            env._get_state(0), env, allowed
        )

        self.assertEqual(action, int(ActionType.ROLL_DICE))
        self.assertEqual(neural_mask, tuple(allowed))

    def test_episode_stores_contiguous_masked_transitions(self) -> None:
        class FakeEnv:
            max_rounds = 1

            def reset(inner_self):
                inner_self.index = 0
                inner_self.done = False
                inner_self.players = [
                    SimpleNamespace(bankrupt=False, properties=[])
                    for _ in range(4)
                ]
                return np.array([0.0], dtype=np.float32)

            def whose_turn(inner_self):
                return (0, 1, 0)[inner_self.index]

            def get_allowed_actions(inner_self, _pid):
                return [int(ActionType.END_TURN)]

            def step(inner_self, _action):
                inner_self.index += 1
                inner_self.done = inner_self.index == 3
                return (
                    np.array([float(inner_self.index)], dtype=np.float32),
                    0.0,
                    inner_self.done,
                    {},
                )

            def winner(inner_self):
                return 0

            def _compute_reward(inner_self, _pid):
                return float(inner_self.index)

        class FakeDDQN:
            player_id = 0
            gamma = 0.99
            win_loss_bonus = 10.0
            decision_penalty = 0.25

            def __init__(inner_self):
                inner_self.transitions = []

            def choose_training_action(inner_self, _state, _env, allowed):
                return allowed[0], tuple(allowed)

            def store_transition(inner_self, *transition):
                inner_self.transitions.append(transition)

            def update(inner_self):
                return {}

            def finish_episode(inner_self):
                inner_self.finished = True

        class FixedAgent:
            player_id = 1

            def choose_action(inner_self, _env):
                return int(ActionType.END_TURN)

        learner = FakeDDQN()
        run_episode(FakeEnv(), learner, [FixedAgent()], 0, is_ppo=False)

        first, terminal = learner.transitions
        np.testing.assert_array_equal(first[0], np.array([0.0]))
        np.testing.assert_array_equal(first[3], np.array([2.0]))
        self.assertEqual(first[5], (int(ActionType.END_TURN),))
        self.assertFalse(first[4])
        self.assertAlmostEqual(first[2], 1.73)
        self.assertTrue(terminal[4])
        self.assertEqual(terminal[5], ())
        self.assertAlmostEqual(terminal[2], 8.0)
        self.assertTrue(learner.finished)

    def test_checkpoint_restores_replay_and_optimizer(self) -> None:
        agent = DDQNAgent(
            0,
            hybrid=True,
            hidden_dim=16,
            batch_size=2,
            buffer_capacity=4,
            device="cpu",
        )
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for action in (int(ActionType.END_TURN), int(ActionType.ROLL_DICE)):
            agent.store_transition(
                state,
                action,
                0.5,
                state,
                False,
                (int(ActionType.END_TURN),),
            )
        self.assertIn("loss", agent.update())
        agent.finish_episode()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ddqn.pt"
            agent.save(str(path))
            restored = DDQNAgent(
                0,
                hybrid=True,
                hidden_dim=16,
                batch_size=2,
                buffer_capacity=4,
                device="cpu",
            )
            restored.load(str(path))

        self.assertEqual(restored.step_count, agent.step_count)
        self.assertEqual(restored.games_trained, agent.games_trained)
        self.assertEqual(restored.exploration_mode, agent.exploration_mode)
        self.assertEqual(restored.decision_penalty, agent.decision_penalty)
        self.assertEqual(len(restored.buffer), len(agent.buffer))
        self.assertEqual(restored.buffer.buffer[0][5], agent.buffer.buffer[0][5])
        for expected, actual in zip(
            agent.online_net.parameters(), restored.online_net.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
