from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLM = ROOT / "SLM_HANDMADE_MONOPOLY"
GENERIC = SLM / "Gemma4_12B_15GB_Colab_QLoRA_Test.ipynb"
PRIMARY = SLM / "Gemma4_12B_Monopoly_QLoRA.ipynb"
RUNNER = SLM / "run_colab_pilot.py"


class Gemma4NotebookTests(unittest.TestCase):
    def test_generic_hardware_reference_is_unchanged(self) -> None:
        digest = hashlib.sha256(GENERIC.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "a7e01e9d09ed2be42fe62fb81f6cabc0d8783bcfddd0fb0ac667e06e8603692b",
        )

    def test_primary_notebook_is_valid_and_all_code_cells_compile(self) -> None:
        notebook = json.loads(PRIMARY.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 7)
        for index, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def test_notebook_contains_the_pilot_gates_and_memory_profile(self) -> None:
        notebook = json.loads(PRIMARY.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        required = (
            '"model": "unsloth/gemma-4-12b-it"',
            '"max_seq_length": 512',
            '"teacher_eval_games": 600',
            '"teacher_min_win_rate": 0.35',
            'needs_evaluation = agent.games_trained >= next_target',
            'gate_games + CONFIG["teacher_increment_games"]',
            'games = next_target - agent.games_trained',
            '"rollouts_per_action": 4',
            '"rollout_horizon": 256',
            '"candidate_limit": 16',
            '"relabel_margin": 0.05',
            '"train_rows": 2048',
            '"validation_rows": 256',
            '"test_rows": 256',
            '"lora_r": 4',
            '"lora_alpha": 4',
            '"micro_batch": 1',
            '"gradient_accumulation": 8',
            '"eval_save_steps": 64',
            'load_in_4bit=True',
            'finetune_attention_modules=True',
            'finetune_mlp_modules=False',
            'dataset_kwargs={"skip_prepare_dataset": True}',
            'processing_class=tokenizer',
            'resume_from_checkpoint=latest',
            'offline["parseable_rate"] >= 0.98',
            'offline["legal_rate"] >= 0.97',
            'offline["exact_rate"] >= 0.65',
            'game_metrics["gemma_win_rate"] >= 0.25',
        )
        for value in required:
            self.assertIn(value, source)
        self.assertIn('userdata.get("HF_TOKEN")', source)
        self.assertNotIn('os.getenv("HF_TOKEN")', source)
        self.assertIn('RUN_ROOT = Path("/content/pilot_v1")', source)
        self.assertNotIn('drive.mount(', source)

    def test_launcher_has_hard_guards_secret_free_archive_and_final_cleanup(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for value in (
            '"git",\n            "archive"',
            'member.name.endswith("/.env")',
            '"--gpu", "T4"',
            '"colab", "ls"',
            '"colab", "exec"',
            '"colab", "download"',
            '"colab", "stop"',
            'finally:',
            'min_ram_gib: float = 1.0',
            'min_ram_gib=0.5',
            'monitor_ram=False',
            'os.killpg(process.pid, signal.SIGTERM)',
            'host_guard(min_ram_gib=min_ram_gib)',
        ):
            self.assertIn(value, source)

    def test_launcher_rejects_colab_notebooks_with_hidden_cell_errors(self) -> None:
        spec = importlib.util.spec_from_file_location("colab_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        notebook = {
            "cells": [{
                "cell_type": "code",
                "outputs": [{
                    "output_type": "error",
                    "ename": "RuntimeError",
                    "evalue": "drive unavailable",
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            with self.assertRaisesRegex(runner.GuardFailure, "drive unavailable"):
                runner.validate_executed_notebook(path)

    def test_launcher_validates_downloaded_resume_snapshots(self) -> None:
        spec = importlib.util.spec_from_file_location("colab_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            safe = Path(directory) / "safe.tar.gz"
            payload = Path(directory) / "manifest.json"
            payload.write_text("{}", encoding="utf-8")
            import tarfile

            with tarfile.open(safe, "w:gz") as archive:
                archive.add(payload, arcname="pilot_v1/manifest.json")
            runner.validate_snapshot(safe)

            missing_manifest = Path(directory) / "missing-manifest.tar.gz"
            with tarfile.open(missing_manifest, "w:gz") as archive:
                archive.add(payload, arcname="pilot_v1/metrics.json")
            with self.assertRaisesRegex(runner.GuardFailure, "manifest.json"):
                runner.validate_snapshot(missing_manifest)

            unsafe = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(unsafe, "w:gz") as archive:
                archive.add(payload, arcname="pilot_v1/.env")
            with self.assertRaisesRegex(runner.GuardFailure, "Unsafe"):
                runner.validate_snapshot(unsafe)


if __name__ == "__main__":
    unittest.main()
