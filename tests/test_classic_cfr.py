from __future__ import annotations

import gzip
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr import (
    CFRConfig,
    MonteCarloCFR,
    SharedGame,
)


class ClassicCFRTests(unittest.TestCase):
    def test_shared_game_clones_preserve_private_dice(self) -> None:
        game = SharedGame.new(seed=9, max_rounds=2)
        clone = game.clone()
        action = game.legal_actions()[0]
        game.step(action)
        clone.step(action)
        roll = game.legal_actions()[0]
        game.step(roll)
        clone.step(roll)
        self.assertEqual(game.env.last_dice, clone.env.last_dice)
        self.assertEqual(game.env.players[game.env.active_player_id()].position,
                         clone.env.players[clone.env.active_player_id()].position)

    def test_optimize_evaluates_every_legal_action(self) -> None:
        game = SharedGame.new(seed=3, max_rounds=2)
        player_id = game.env.whose_turn()
        prop = game.env.properties[1]
        prop.owner = player_id
        game.env.players[player_id].properties.append(prop)
        game.env._update_monopolies()
        model = MonteCarloCFR(
            CFRConfig(sims_per_action=1, rollout_horizon=1, max_rounds=2)
        )
        legal = game.legal_actions()
        self.assertGreater(len(legal), 1)
        with patch.object(model, "_evaluate_action", return_value=0.25) as evaluate:
            action, metrics = model.optimize(game, game_index=0, decision_index=0)
        self.assertIn(action, legal)
        self.assertEqual(evaluate.call_count, len(legal))
        self.assertEqual(metrics["actions"], float(len(legal)))
        self.assertEqual(len(model.tables[player_id]), 1)

    def test_checkpoint_round_trip(self) -> None:
        game = SharedGame.new(seed=4, max_rounds=2)
        model = MonteCarloCFR(CFRConfig(rollout_horizon=1, max_rounds=2))
        with patch.object(model, "_evaluate_action", return_value=0.0):
            model.optimize(game, game_index=0, decision_index=0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfr.pkl.gz"
            model.save(path)
            restored = MonteCarloCFR.load(path)
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)

        self.assertEqual(restored.config, model.config)
        self.assertEqual(payload["format_version"], 3)
        saved_node = next(
            value for table in payload["tables"] for value in table.values()
        )
        self.assertIsInstance(saved_node, tuple)
        self.assertEqual(
            [len(table) for table in restored.tables],
            [len(table) for table in model.tables],
        )

    def test_train_game_writes_decision_checkpoint(self) -> None:
        model = MonteCarloCFR(
            CFRConfig(rollout_horizon=1, max_rounds=1, max_decisions=2)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pkl.gz"
            model.train_game(
                checkpoint_every_decisions=1,
                checkpoint_path=path,
            )

            self.assertTrue(path.exists())
            restored = MonteCarloCFR.load(path)
            self.assertGreater(sum(map(len, restored.tables)), 0)
            self.assertIsNotNone(restored.in_progress)
            self.assertEqual(restored.in_progress["decisions"], 2)
            self.assertEqual(restored.games_trained, 0)

            with self.assertRaisesRegex(ValueError, "increase max_decisions"):
                restored.train_game()

    def test_interrupted_game_resumes_same_trajectory(self) -> None:
        class StopAfterTwo:
            def __init__(self):
                self.calls = 0

            def check(self):
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("stop")

        model = MonteCarloCFR(
            CFRConfig(rollout_horizon=1, max_rounds=2, max_decisions=4)
        )
        with self.assertRaisesRegex(RuntimeError, "stop"):
            model.train_game(watchdog=StopAfterTwo())

        self.assertEqual(model.in_progress["decisions"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pkl.gz"
            model.save(path)
            restored = MonteCarloCFR.load(path)
            stats = restored.train_game()

        self.assertTrue(stats["resumed"])
        self.assertEqual(stats["decisions"], 4)
        self.assertTrue(stats["truncated"])
        self.assertEqual(restored.games_trained, 0)


if __name__ == "__main__":
    unittest.main()
