#!/usr/bin/env python3
"""Run the staged Gemma Monopoly pilot on a named Colab T4 session."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - launcher host normally has psutil
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).with_name("Gemma4_12B_Monopoly_QLoRA.ipynb")
STAGES = ("teacher", "collect", "train", "eval")
STAGE_TIMEOUT_HOURS = {"teacher": 14, "collect": 20, "train": 12, "eval": 20}
DRIVE_OUTPUT = (
    "/content/drive/MyDrive/DeepRL_Monopoly/Gemma4_12B_Monopoly/"
    "pilot_v1/notebook_output"
)


class GuardFailure(RuntimeError):
    pass


def host_guard(min_ram_gib: float = 2.0, min_disk_gib: float = 10.0) -> None:
    if psutil is not None and psutil.virtual_memory().available < min_ram_gib * 1024**3:
        raise GuardFailure("Laptop available RAM fell below the 2 GiB guard")
    free = shutil.disk_usage(ROOT).free
    if free < min_disk_gib * 1024**3:
        raise GuardFailure("Laptop free disk fell below the 10 GiB guard")


def run_guarded(command: list[str], timeout: float, cwd: Path = ROOT) -> None:
    host_guard()
    print("+", shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Hard timeout reached for {command[0]}")
            host_guard()
            time.sleep(2)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def require_committed_pipeline() -> None:
    subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=True)
    for path in (NOTEBOOK, Path(__file__)):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )


def build_archive(destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            "--prefix=DeepRL_Monopoly/",
            f"--output={temporary}",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    with tarfile.open(temporary, "r:gz") as archive:
        unsafe = [
            member.name
            for member in archive.getmembers()
            if member.name.endswith("/.env")
            or "/__pycache__/" in member.name
            or member.name.endswith(".pyc")
        ]
    if unsafe:
        temporary.unlink(missing_ok=True)
        raise GuardFailure(f"Secret/cache safety check failed: {unsafe[:3]}")
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="gemma4-monopoly-pilot-v1")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("colab") is None:
        raise SystemExit("Google Colab CLI is not installed")
    require_committed_pipeline()
    artifact_dir = ROOT / "artifacts" / "gemma4_monopoly_colab"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def stop_on_signal(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)

    with tempfile.TemporaryDirectory(prefix="gemma4-monopoly-") as temporary_name:
        temporary = Path(temporary_name)
        archive = temporary / "DeepRL_Monopoly.tar.gz"
        build_archive(archive)
        if args.dry_run:
            print(f"Dry run passed; secret-free archive size: {archive.stat().st_size} bytes")
            return 0

        cleanup_needed = True
        try:
            run_guarded(["colab", "version"], timeout=60)
            run_guarded(
                ["colab", "new", "-s", args.session, "--gpu", "T4"],
                timeout=20 * 60,
            )
            run_guarded(
                ["colab", "drivemount", "-s", args.session],
                timeout=20 * 60,
            )
            run_guarded(
                [
                    "colab", "upload", "-s", args.session,
                    str(archive), "/content/DeepRL_Monopoly.tar.gz",
                ],
                timeout=20 * 60,
            )

            for stage in args.stages:
                stage_file = temporary / "monopoly_stage.txt"
                stage_file.write_text(stage + "\n", encoding="utf-8")
                run_guarded(
                    [
                        "colab", "upload", "-s", args.session,
                        str(stage_file), "/content/monopoly_stage.txt",
                    ],
                    timeout=5 * 60,
                )
                stage_notebook = temporary / f"Gemma4_12B_Monopoly_QLoRA_{stage}.ipynb"
                shutil.copy2(NOTEBOOK, stage_notebook)
                timeout_seconds = STAGE_TIMEOUT_HOURS[stage] * 3600
                run_guarded(
                    [
                        "colab", "exec", "-s", args.session,
                        "-f", str(stage_notebook), "--timeout", str(timeout_seconds),
                    ],
                    timeout=timeout_seconds + 5 * 60,
                )
                executed = stage_notebook.with_name(stage_notebook.stem + "_output.ipynb")
                if not executed.exists():
                    raise GuardFailure(f"Colab did not export the executed {stage} notebook")
                local_executed = artifact_dir / executed.name
                shutil.copy2(executed, local_executed)
                log_path = artifact_dir / f"{stage}_execution.md"
                run_guarded(
                    ["colab", "log", "-s", args.session, "-o", str(log_path)],
                    timeout=10 * 60,
                )
                for local_path in (local_executed, log_path):
                    run_guarded(
                        [
                            "colab", "upload", "-s", args.session,
                            str(local_path), f"{DRIVE_OUTPUT}/{local_path.name}",
                        ],
                        timeout=20 * 60,
                    )
        finally:
            if cleanup_needed:
                try:
                    subprocess.run(
                        ["colab", "stop", "-s", args.session],
                        cwd=ROOT,
                        timeout=5 * 60,
                        check=False,
                    )
                except Exception as exc:  # best effort after the hard cleanup path
                    print(f"WARNING: Colab session termination failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
