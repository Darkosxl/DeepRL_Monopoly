import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "train_and_save.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("train_and_save", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingHistoryTests(unittest.TestCase):
    def test_resume_merges_series_and_resource_totals(self):
        previous = {
            "games": [40, 80],
            "win_rates": [0.0, 2.5],
            "resumed_from_games": 0,
            "games_completed_this_run": 100,
            "games_completed": 100,
            "elapsed_seconds": 200.0,
            "peak_rss_gib": 1.2,
            "peak_cuda_gib": 0.04,
        }
        current = {
            "games": [140, 180],
            "win_rates": [5.0, 7.5],
            "resumed_from_games": 100,
            "games_completed_this_run": 100,
            "games_completed": 200,
            "elapsed_seconds": 180.0,
            "seconds_per_game": 1.8,
            "peak_rss_gib": 1.1,
            "peak_cuda_gib": 0.05,
        }

        merged = MODULE.merge_training_history(previous, current)

        self.assertEqual(merged["games"], [40, 80, 140, 180])
        self.assertEqual(merged["win_rates"], [0.0, 2.5, 5.0, 7.5])
        self.assertEqual(merged["games_completed"], 200)
        self.assertEqual(merged["games_completed_this_run"], 200)
        self.assertEqual(merged["elapsed_seconds"], 380.0)
        self.assertEqual(merged["seconds_per_game"], 1.9)
        self.assertEqual(merged["peak_rss_gib"], 1.2)
        self.assertEqual(merged["peak_cuda_gib"], 0.05)
        self.assertEqual(len(merged["training_segments"]), 2)

    def test_resume_rejects_mismatched_history(self):
        with self.assertRaisesRegex(ValueError, "History/checkpoint mismatch"):
            MODULE.merge_training_history(
                {"games_completed": 99}, {"resumed_from_games": 100}
            )

    def test_resume_without_log_window_preserves_previous_series(self):
        previous = {
            "games": [40],
            "win_rates": [0.0],
            "games_completed": 40,
            "games_completed_this_run": 40,
        }
        current = {
            "resumed_from_games": 40,
            "games_completed": 41,
            "games_completed_this_run": 1,
        }

        merged = MODULE.merge_training_history(previous, current)

        self.assertEqual(merged["games"], [40])
        self.assertEqual(merged["win_rates"], [0.0])


if __name__ == "__main__":
    unittest.main()
