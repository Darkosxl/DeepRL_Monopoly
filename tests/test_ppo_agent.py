from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = ROOT / "RL_PPO(UNOFFICIAL)_MONOPOLY"
sys.path.insert(0, str(PPO_ROOT))

from monopoly_drl.agent_ppo import PPOAgent  # noqa: E402
from monopoly_drl.env import MonopolyEnv  # noqa: E402


class PPOAgentTests(unittest.TestCase):
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
            for expected, actual in zip(
                agent.actor.parameters(), restored.actor.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

    def test_legacy_checkpoint_has_clear_error(self) -> None:
        legacy = PPO_ROOT / "hy_model.pt"
        if not legacy.exists():
            self.skipTest("legacy checkpoint is unavailable")
        with self.assertRaisesRegex(ValueError, "Legacy PPO checkpoint"):
            PPOAgent(0, hybrid=True, device="cpu").load(str(legacy))


if __name__ == "__main__":
    unittest.main()
