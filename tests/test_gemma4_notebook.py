from __future__ import annotations

import hashlib
import json
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

    def test_launcher_has_hard_guards_secret_free_archive_and_final_cleanup(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for value in (
            '"git",\n            "archive"',
            'member.name.endswith("/.env")',
            '"--gpu", "T4"',
            '"colab", "drivemount"',
            '"colab", "exec"',
            '"colab", "stop"',
            'finally:',
            'os.killpg(process.pid, signal.SIGTERM)',
            'host_guard()',
        ):
            self.assertIn(value, source)


if __name__ == "__main__":
    unittest.main()
