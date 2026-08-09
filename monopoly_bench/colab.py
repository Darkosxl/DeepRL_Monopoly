"""Secret-free Colab T4 handoff with unconditional named-session cleanup."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Callable

from .engine import ROOT


SESSION_NAME = "monopolyzero-ppo-plus-v2-v1"
REMOTE_RUNNER = Path(__file__).with_name("colab") / "MonopolyZero_Train.ipynb"
DEFAULT_PPO = ROOT / "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"
DEFAULT_CFR = ROOT / "artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz"
REPLAY_FILES = {
    "metadata.json",
    "states.mmap",
    "masks.mmap",
    "actions.mmap",
    "visit_counts.mmap",
    "q_values.mmap",
    "lengths.mmap",
    "selected.mmap",
    "values.mmap",
    "outcomes.mmap",
    "actors.mmap",
    "game_ids.mmap",
}


class ColabHandoffError(RuntimeError):
    pass


def _validate_archive(path: Path, prefix: str | None = None) -> None:
    root = None if prefix is None else prefix.rstrip("/")
    with tarfile.open(path, "r:gz") as archive:
        unsafe = [
            member.name
            for member in archive.getmembers()
            if member.name.startswith("/")
            or ".." in Path(member.name).parts
            or member.name.endswith("/.env")
            or "/__pycache__/" in member.name
            or member.name.endswith(".pyc")
            or (
                prefix is not None
                and member.name != root
                and not member.name.startswith(prefix)
            )
        ]
    if unsafe:
        raise ColabHandoffError(f"Unsafe archive members: {unsafe[:3]}")


def build_git_archive(destination: str | Path) -> Path:
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "--prefix=DeepRL_Monopoly/", f"--output={temporary}", "HEAD"],
        cwd=ROOT,
        check=True,
    )
    _validate_archive(temporary, "DeepRL_Monopoly/")
    os.replace(temporary, destination)
    return destination


def build_baseline_bundle(
    destination: str | Path,
    bootstrap_ppo: str | Path = DEFAULT_PPO,
    cfr_checkpoint: str | Path = DEFAULT_CFR,
) -> Path:
    destination = Path(destination)
    bootstrap_ppo, cfr_checkpoint = Path(bootstrap_ppo), Path(cfr_checkpoint)
    if not bootstrap_ppo.is_file():
        raise ColabHandoffError(f"PPO bootstrap is unavailable: {bootstrap_ppo}")
    temporary = destination.with_name(destination.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(bootstrap_ppo, arcname="artifacts/ppo_plus/ppo_hybrid_2000_v2.pt", recursive=False)
        if cfr_checkpoint.is_file():
            archive.add(cfr_checkpoint, arcname="artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz", recursive=False)
    _validate_archive(temporary, "artifacts/")
    os.replace(temporary, destination)
    return destination


def _allowed_resume_path(relative: Path) -> bool:
    if relative.as_posix() in {"config.json", "status.json", "asu_expert.npz"}:
        return True
    if len(relative.parts) == 2:
        directory, name = relative.parts
        if directory in {"checkpoints", "snapshots", "candidates"}:
            return name.endswith(".pt")
        if directory == "reports":
            return name.endswith(".json")
        if directory == "replay":
            return name in REPLAY_FILES
    return (
        len(relative.parts) == 3
        and relative.parts[0] == "pending"
        and relative.parts[1].startswith("generation_")
        and relative.parts[2] in REPLAY_FILES
    )


def build_resume_bundle(run_dir: str | Path, destination: str | Path) -> Path:
    run_dir, destination = Path(run_dir), Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {destination.resolve(), temporary.resolve()}
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            relative = path.relative_to(run_dir)
            if path.is_dir() or path.resolve() in excluded or not _allowed_resume_path(relative):
                continue
            if path.is_symlink():
                raise ColabHandoffError(f"Resume bundle refuses symlink: {relative}")
            archive.add(path, arcname=str(Path("run") / relative), recursive=False)
    _validate_archive(temporary, "run/")
    os.replace(temporary, destination)
    return destination


class ColabLauncher:
    def __init__(
        self,
        session: str = SESSION_NAME,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.session = session
        self.runner = runner

    def _run(self, command: list[str], *, check: bool = True) -> None:
        self.runner(command, cwd=ROOT, check=check)

    def launch(
        self,
        run_dir: str | Path,
        *,
        bootstrap_ppo: str | Path = DEFAULT_PPO,
        cfr_checkpoint: str | Path = DEFAULT_CFR,
        dry_run: bool = False,
    ) -> Path | None:
        if not dry_run and shutil.which("colab") is None:
            raise ColabHandoffError("Google Colab CLI is not installed")
        run_dir = Path(run_dir)
        output = run_dir / "colab" / "result.bundle"
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="monopolyzero-colab-") as temporary_name:
            temporary = Path(temporary_name)
            archive = build_git_archive(temporary / "DeepRL_Monopoly.tar.gz")
            bundle = build_resume_bundle(run_dir, temporary / "resume.bundle")
            baselines = build_baseline_bundle(
                temporary / "baselines.tar.gz",
                bootstrap_ppo,
                cfr_checkpoint,
            )
            if dry_run:
                return None
            try:
                self._run(["colab", "new", "-s", self.session, "--gpu", "T4"])
                self._run(["colab", "upload", "-s", self.session, str(archive), "/content/DeepRL_Monopoly.tar.gz"])
                self._run(["colab", "upload", "-s", self.session, str(bundle), "/content/monopolyzero-resume.tar.gz"])
                self._run(["colab", "upload", "-s", self.session, str(baselines), "/content/monopolyzero-baselines.tar.gz"])
                execution_error = None
                try:
                    self._run(["colab", "exec", "-s", self.session, "-f", str(REMOTE_RUNNER), "--timeout", "86400"])
                except Exception as exc:
                    execution_error = exc
                temporary_output = output.with_name(output.name + ".tmp")
                self._run(["colab", "download", "-s", self.session, "/content/monopolyzero-result.tar.gz", str(temporary_output)])
                _validate_archive(temporary_output, "run/")
                os.replace(temporary_output, output)
                if execution_error is not None:
                    raise execution_error
                return output
            finally:
                try:
                    self._run(["colab", "stop", "-s", self.session], check=False)
                except Exception:
                    pass
