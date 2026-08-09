from __future__ import annotations

import json
import shutil
import tarfile

import numpy as np
import pytest

from monopoly_bench.colab import ColabLauncher, build_resume_bundle
from monopoly_bench.config import BenchmarkConfig, ResourceConfig, SearchConfig
from monopoly_bench.exports import export_teacher
from monopoly_bench.model import MonopolyZeroNet
from monopoly_bench.training import Trainer
from monopoly_bench.watchdog import GIB, ResourceLimitReached, ResourceSnapshot, ResourceWatchdog


PPO = "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"


def test_native_and_canonical_slm_exports(bench_tmp) -> None:
    champion = bench_tmp / "champion.pt"
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    model.save_inference(champion)
    config = BenchmarkConfig(max_rounds=1, search=SearchConfig(simulations=1, max_depth=8))
    manifest = export_teacher(
        champion,
        games=2,
        output_dir=bench_tmp / "teacher",
        config=config,
        shard_positions=32,
    )
    assert manifest["native_positions"] > 0
    assert manifest["slm_rows"]["validation"] <= 64
    assert manifest["slm_rows"]["test"] <= 64
    shard = np.load(bench_tmp / "teacher" / manifest["shards"][0]["path"])
    assert {"states", "legal_masks", "visit_counts", "q_values", "selected_actions", "values", "outcomes"}.issubset(shard.files)
    row = json.loads((bench_tmp / "teacher/slm/validation.jsonl").read_text().splitlines()[0])
    assert row["completion"].startswith('{"action"')
    assert row["teacher_action"] in row["legal_actions"]
    assert len(row["root_value"]) == 4


def test_watchdog_enforces_soft_hard_ram_disk_and_vram(monkeypatch, bench_tmp) -> None:
    watchdog = ResourceWatchdog(bench_tmp, ResourceConfig(min_free_disk_gib=0))
    monkeypatch.setattr(
        watchdog,
        "snapshot",
        lambda: ResourceSnapshot(3 * GIB, 10 * GIB, 30 * GIB, 10 * GIB),
    )
    with pytest.raises(ResourceLimitReached) as error:
        watchdog.check()
    assert error.value.resource == "soft_rss"


def test_colab_failure_always_stops_named_session(monkeypatch, bench_tmp) -> None:
    run_dir = bench_tmp / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(command) > 1 and command[1] == "upload":
            raise RuntimeError("upload failed")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/colab")
    launcher = ColabLauncher(session="test-session", runner=runner)
    with pytest.raises(RuntimeError, match="upload failed"):
        launcher.launch(run_dir)
    assert calls[-1] == ["colab", "stop", "-s", "test-session"]


def test_resume_bundle_is_an_explicit_secret_free_allowlist(bench_tmp) -> None:
    run_dir = bench_tmp / "run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "config.json").write_text("{}")
    (run_dir / "reports/generation.json").write_text("{}")
    (run_dir / ".env.local").write_text("TOKEN=secret")
    (run_dir / "credentials.json").write_text('{"token":"secret"}')
    bundle = build_resume_bundle(run_dir, bench_tmp / "resume.bundle")
    with tarfile.open(bundle, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    assert names == {"run/config.json", "run/reports/generation.json"}


def test_watchdog_handoff_always_writes_a_portable_bundle(bench_tmp) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.run_dir = bench_tmp / "run"
    trainer.run_dir.mkdir()
    trainer.generation = 4
    trainer.fallback_colab = False

    def checkpoint():
        path = trainer.run_dir / "checkpoints/generation_0004.pt"
        path.parent.mkdir()
        path.write_bytes(b"portable checkpoint")
        return path

    trainer._checkpoint = checkpoint
    trainer._handoff("test resource limit")
    assert (trainer.run_dir / "colab/resume_generation_0004.bundle").exists()
    report = json.loads((trainer.run_dir / "reports/handoff.json").read_text())
    assert report["reason"] == "test resource limit"
