from __future__ import annotations

import hashlib
import gzip
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
DRIVE_SYNC = SLM / "drive_checkpoint_sync.py"


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
            '"teacher_policy": "asu_value_v1"',
            '"exploration_every": 9',
            '"exploration_top_k": 3',
            'action, candidates = asu_teacher_decision(',
            'collect_teacher_game(',
            '"asu_teacher_hash": ASU_TEACHER_HASH',
            '"candidate_limit": 16',
            '"min_recovery_fraction": 0.25',
            'COLLECTION_GAMES_PER_SESSION = 8',
            'len(progress["games"]) % COLLECTION_GAMES_PER_SESSION == 0',
            '"collect", fingerprint, "partial"',
            'Collection checkpoint ready; resume the collect stage.',
            'Archived incompatible collection checkpoint',
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
            'game_metrics["asu_win_rate"] - 0.10',
        )
        for value in required:
            self.assertIn(value, source)
        self.assertIn('userdata.get("HF_TOKEN")', source)
        self.assertIn('HF_TOKEN = get_token()', source)
        self.assertNotIn('os.getenv("HF_TOKEN")', source)
        self.assertIn(
            "from unsloth import FastModel, is_bfloat16_supported\n"
            "    from datasets import Dataset\n"
            "    from transformers import TrainerCallback, default_data_collator",
            source,
        )
        self.assertIn('RUN_ROOT = Path("/content/asu_pilot_v1")', source)
        self.assertNotIn('drive.mount(', source)
        self.assertNotIn('from monopoly_game_engine.agent_ppo import PPOAgent', source)
        self.assertNotIn('ppo_plus_v2_teacher.pt', source)
        self.assertNotIn('teacher_checkpoint_hash', source)
        self.assertNotIn('heuristic_teacher(', source)
        self.assertIn('DRIVE_ADAPTER_DIR / "checkpoints"', source)
        self.assertIn(
            'checkpoint_fingerprint = checkpoint_dir / "checkpoint_fingerprint.json"',
            source,
        )
        self.assertIn('Google Drive artifact hash mismatch', source)

    def test_launcher_has_hard_guards_secret_free_archive_and_final_cleanup(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for value in (
            '"git",\n            "archive"',
            'member.name.endswith("/.env")',
            '"--gpu", "T4"',
            '"colab", "ls"',
            '"colab", "exec"',
            '"colab", "download"',
            '"colab", "drivemount"',
            '"colab", "stop"',
            'finally:',
            'min_ram_gib: float = 1.0',
            'min_ram_gib=0.5',
            'monitor_ram=False',
            'os.killpg(process.pid, signal.SIGTERM)',
            'host_guard(min_ram_gib=min_ram_gib)',
            'RUN_NAME = "asu_pilot_v1"',
            '"--drive-checkpoints"',
            '"--hf-token-file"',
            '"--rclone-config"',
            '"--rclone-binary"',
            '"/content/.hf_token"',
            'Hugging Face authentication ready.',
            'Google Drive rclone checkpoint sync started:',
            'Google Drive training artifacts verified uploaded.',
            '/ "gemma4_monopoly_colab" / RUN_NAME',
            'ROOT / "SLM_HANDMADE_MONOPOLY" / "monopoly_qlora.py"',
            'ROOT / "ASU_FROZEN_TEACHER" / "core.py"',
            'ROOT / "ASU_FROZEN_TEACHER" / "spec.py"',
            'ROOT / "ASU_FROZEN_TEACHER" / "types.py"',
        ):
            self.assertIn(value, source)

        spec = importlib.util.spec_from_file_location("colab_runner_paths", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(runner.ARTIFACT_ROOT.name, "asu_pilot_v1")
        self.assertTrue(
            {"monopoly_qlora.py", "drive_checkpoint_sync.py", "core.py", "spec.py", "types.py"}
            <= {path.name for path in runner.PIPELINE_FILES}
        )

    def test_drive_checkpoint_sync_compiles_and_has_transport_checks(self) -> None:
        source = DRIVE_SYNC.read_text(encoding="utf-8")
        compile(source, str(DRIVE_SYNC), "exec")
        for value in (
            'RCLONE_CONFIG.unlink(missing_ok=True)',
            'adapter_model.safetensors',
            '"--checksum"',
            '"sha256": sha256(archive)',
            'drive_sync_complete.json',
        ):
            self.assertIn(value, source)

    def test_launcher_validates_private_rclone_inputs(self) -> None:
        spec = importlib.util.spec_from_file_location("colab_runner_rclone", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "rclone.conf"
            binary = Path(directory) / "rclone"
            config.write_text("[gemma_drive]\ntype = drive\n", encoding="utf-8")
            config.chmod(0o600)
            binary.write_bytes(b"binary")
            self.assertEqual(
                runner.validate_rclone_inputs(config, binary),
                (config.resolve(), binary.resolve()),
            )
            config.chmod(0o644)
            with self.assertRaisesRegex(runner.GuardFailure, "group or other"):
                runner.validate_rclone_inputs(config, binary)

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
                archive.add(payload, arcname="asu_pilot_v1/manifest.json")
            runner.validate_snapshot(safe)

            missing_manifest = Path(directory) / "missing-manifest.tar.gz"
            with tarfile.open(missing_manifest, "w:gz") as archive:
                archive.add(payload, arcname="asu_pilot_v1/metrics.json")
            with self.assertRaisesRegex(runner.GuardFailure, "manifest.json"):
                runner.validate_snapshot(missing_manifest)

            unsafe = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(unsafe, "w:gz") as archive:
                archive.add(payload, arcname="asu_pilot_v1/.env")
            with self.assertRaisesRegex(runner.GuardFailure, "Unsafe"):
                runner.validate_snapshot(unsafe)

    def test_launcher_validates_local_collection_progress(self) -> None:
        spec = importlib.util.spec_from_file_location("colab_runner_progress", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection_progress.json"
            path.write_text(json.dumps({
                "fingerprint": "old",
                "next_seed": 2,
                "rows": [{"state_hash": "x"}],
                "games": [{"game_id": "g"}],
            }), encoding="utf-8")
            runner.validate_collection_progress(path)
            compressed = Path(directory) / "collection_progress.json.gz"
            runner.compress_collection_progress(path, compressed)
            with gzip.open(compressed, "rt", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["next_seed"], 2)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(runner.GuardFailure, "invalid schema"):
                runner.validate_collection_progress(path)
            with self.assertRaisesRegex(runner.GuardFailure, "invalid schema"):
                runner.compress_collection_progress(path, compressed)


if __name__ == "__main__":
    unittest.main()
