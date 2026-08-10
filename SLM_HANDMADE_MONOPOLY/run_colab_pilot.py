#!/usr/bin/env python3
"""Run the staged Gemma Monopoly pilot on a named Colab T4 session."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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
DRIVE_SYNC = Path(__file__).with_name("drive_checkpoint_sync.py")
RUN_NAME = "asu_pilot_v1"
REMOTE_RUN_ROOT = f"/content/{RUN_NAME}"
ARTIFACT_ROOT = ROOT / "artifacts" / "gemma4_monopoly_colab" / RUN_NAME
DRIVE_CHECKPOINT_ROOT = (
    "gemma_drive:DeepRL_Monopoly/Gemma4_12B_Monopoly/"
    "pilot_v1/adapters/checkpoints"
)
STAGES = ("teacher", "collect", "train", "eval")
STAGE_TIMEOUT_HOURS = {"teacher": 1, "collect": 20, "train": 12, "eval": 20}
PIPELINE_FILES = (
    NOTEBOOK,
    DRIVE_SYNC,
    Path(__file__),
    ROOT / "SLM_HANDMADE_MONOPOLY" / "monopoly_qlora.py",
    ROOT / "ASU_FROZEN_TEACHER" / "__init__.py",
    ROOT / "ASU_FROZEN_TEACHER" / "core.py",
    ROOT / "ASU_FROZEN_TEACHER" / "spec.py",
    ROOT / "ASU_FROZEN_TEACHER" / "types.py",
)


class GuardFailure(RuntimeError):
    pass


def host_guard(min_ram_gib: float = 1.0, min_disk_gib: float = 10.0) -> None:
    if psutil is not None and psutil.virtual_memory().available < min_ram_gib * 1024**3:
        raise GuardFailure(
            f"Laptop available RAM fell below the {min_ram_gib:g} GiB guard"
        )
    free = shutil.disk_usage(ROOT).free
    if free < min_disk_gib * 1024**3:
        raise GuardFailure("Laptop free disk fell below the 10 GiB guard")


def run_guarded(
    command: list[str],
    timeout: float,
    cwd: Path = ROOT,
    *,
    min_ram_gib: float = 1.0,
    monitor_ram: bool = True,
) -> None:
    host_guard(min_ram_gib=min_ram_gib)
    print("+", shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Hard timeout reached for {command[0]}")
            host_guard(min_ram_gib=min_ram_gib if monitor_ram else 0.0)
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
    for path in PIPELINE_FILES:
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


def validate_executed_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors = [
        output
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        first = errors[0]
        raise GuardFailure(
            f"Notebook execution failed with {first.get('ename')}: "
            f"{first.get('evalue')} ({len(errors)} error cell(s))"
        )


def validate_snapshot(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = {member.name.rstrip("/") for member in archive.getmembers()}
        unsafe = [
            member.name
            for member in archive.getmembers()
            if (
                (member.name != RUN_NAME and not member.name.startswith(f"{RUN_NAME}/"))
                or member.name.endswith("/.env")
                or "/__pycache__/" in member.name
                or member.name.endswith(".pyc")
            )
        ]
    if unsafe:
        raise GuardFailure(f"Unsafe pilot snapshot members: {unsafe[:3]}")
    if f"{RUN_NAME}/manifest.json" not in names:
        raise GuardFailure("Pilot snapshot has no manifest.json")


def snapshot_run(
    session: str,
    temporary: Path,
    artifact_dir: Path,
    label: str,
    *,
    min_ram_gib: float = 1.0,
) -> Path:
    remote_snapshot = f"/content/{RUN_NAME}_snapshot_{time.time_ns()}.tar.gz"
    script = temporary / "snapshot_pilot.py"
    script.write_text(
        "from pathlib import Path\n"
        "import os, tarfile\n"
        f"root = Path({REMOTE_RUN_ROOT!r})\n"
        "if not (root / 'manifest.json').exists():\n"
        "    raise RuntimeError('No resumable pilot manifest exists')\n"
        f"target = Path({remote_snapshot!r})\n"
        "temporary = target.with_name(target.name + '.tmp')\n"
        "with tarfile.open(temporary, 'w:gz') as archive:\n"
        f"    archive.add(root, arcname={RUN_NAME!r})\n"
        "os.replace(temporary, target)\n",
        encoding="utf-8",
    )
    run_guarded(
        ["colab", "exec", "-s", session, "-f", str(script), "--timeout", "3600"],
        timeout=3700,
        min_ram_gib=min_ram_gib,
        monitor_ram=False,
    )
    downloaded = artifact_dir / f"{RUN_NAME}_{label}.tar.gz.tmp"
    downloaded.unlink(missing_ok=True)
    try:
        run_guarded(
            ["colab", "download", "-s", session, remote_snapshot, str(downloaded)],
            timeout=3700,
            min_ram_gib=min_ram_gib,
        )
        validate_snapshot(downloaded)
    except BaseException:
        downloaded.unlink(missing_ok=True)
        raise
    finally:
        try:
            run_guarded(
                ["colab", "rm", "-s", session, remote_snapshot],
                timeout=5 * 60,
                min_ram_gib=min_ram_gib,
            )
        except Exception as exc:
            print(f"WARNING: remote snapshot cleanup failed: {exc}", flush=True)
    stage_snapshot = artifact_dir / f"{RUN_NAME}_{label}.tar.gz"
    os.replace(downloaded, stage_snapshot)
    latest = artifact_dir / f"{RUN_NAME}_latest.tar.gz"
    latest_temporary = latest.with_name(latest.name + ".tmp")
    shutil.copy2(stage_snapshot, latest_temporary)
    os.replace(latest_temporary, latest)
    return stage_snapshot


def restore_snapshot(session: str, temporary: Path, artifact_dir: Path) -> None:
    latest = artifact_dir / f"{RUN_NAME}_latest.tar.gz"
    if not latest.exists():
        return
    validate_snapshot(latest)
    run_guarded(
        [
            "colab", "upload", "-s", session,
            str(latest), f"/content/{RUN_NAME}_resume.tar.gz",
        ],
        timeout=3700,
    )
    script = temporary / "restore_pilot.py"
    script.write_text(
        "from pathlib import Path\n"
        "import tarfile\n"
        f"snapshot = Path('/content/{RUN_NAME}_resume.tar.gz')\n"
        "with tarfile.open(snapshot, 'r:gz') as archive:\n"
        "    archive.extractall('/content', filter='data')\n"
        "snapshot.unlink()\n",
        encoding="utf-8",
    )
    run_guarded(
        ["colab", "exec", "-s", session, "-f", str(script), "--timeout", "3600"],
        timeout=3700,
        monitor_ram=False,
    )
    run_guarded(
        ["colab", "ls", "-s", session, f"{REMOTE_RUN_ROOT}/manifest.json"],
        timeout=5 * 60,
    )


def validate_collection_progress_data(progress) -> None:
    required = {"fingerprint", "next_seed", "rows", "games"}
    if not isinstance(progress, dict) or required - progress.keys():
        raise GuardFailure("Collection progress has an invalid schema")
    if not progress["rows"] or not progress["games"]:
        raise GuardFailure("Collection progress has no resumable games")


def validate_collection_progress(path: Path) -> None:
    if path.name == ".env" or not path.is_file():
        raise GuardFailure("Collection progress path is missing or unsafe")
    with path.open(encoding="utf-8") as handle:
        validate_collection_progress_data(json.load(handle))


def compress_collection_progress(path: Path, destination: Path) -> None:
    if path.name == ".env" or not path.is_file():
        raise GuardFailure("Collection progress path is missing or unsafe")
    pending = destination.with_name(destination.name + ".tmp")
    try:
        with path.open("rb") as source, gzip.open(pending, "wb") as target:
            shutil.copyfileobj(source, target)
        with gzip.open(pending, "rt", encoding="utf-8") as handle:
            validate_collection_progress_data(json.load(handle))
        os.replace(pending, destination)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def upload_collection_progress(session: str, path: Path, temporary: Path) -> None:
    compressed = temporary / f"{RUN_NAME}_collection_progress.json.gz"
    compress_collection_progress(path, compressed)
    run_guarded(
        [
            "colab", "upload", "-s", session, str(compressed),
            f"/content/{RUN_NAME}_collection_progress.json.gz",
        ],
        timeout=3700,
    )


def validate_rclone_inputs(config: Path, binary: Path) -> tuple[Path, Path]:
    config = config.resolve()
    binary = binary.resolve()
    if config.name == ".env" or not config.is_file():
        raise GuardFailure("rclone config path is missing or unsafe")
    if binary.name == ".env" or not binary.is_file() or binary.stat().st_size == 0:
        raise GuardFailure("rclone binary path is missing or unsafe")
    if config.stat().st_mode & 0o077:
        raise GuardFailure("rclone config must not be readable by group or other users")
    if "[gemma_drive]" not in config.read_text(encoding="utf-8"):
        raise GuardFailure("rclone config has no gemma_drive remote")
    return config, binary


def rclone_reference(binary: Path) -> tuple[str, str]:
    result = subprocess.run(
        [str(binary), "version"], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    if not lines or not lines[0].startswith("rclone v"):
        raise GuardFailure("rclone binary did not report a version")
    version = lines[0].removeprefix("rclone v")
    if not version or any(not part.isdigit() for part in version.split(".")):
        raise GuardFailure("rclone binary version is unsafe")
    with binary.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return version, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="gemma4-monopoly-asu-v1")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--collection-progress", type=Path)
    parser.add_argument("--drive-checkpoints", action="store_true")
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--rclone-config", type=Path)
    parser.add_argument("--rclone-binary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.rclone_config is None) != (args.rclone_binary is None):
        raise SystemExit("--rclone-config and --rclone-binary must be supplied together")
    if args.drive_checkpoints and args.rclone_config:
        raise SystemExit("Choose DriveFS or rclone checkpoints, not both")
    if shutil.which("colab") is None:
        raise SystemExit("Google Colab CLI is not installed")
    require_committed_pipeline()
    artifact_dir = ARTIFACT_ROOT
    artifact_dir.mkdir(parents=True, exist_ok=True)

    interrupted = False

    def stop_on_signal(signum, _frame):
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
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
        snapshot_needed = False
        try:
            run_guarded(["colab", "version"], timeout=60)
            run_guarded(
                ["colab", "new", "-s", args.session, "--gpu", "T4"],
                timeout=20 * 60,
            )
            if args.drive_checkpoints:
                run_guarded(
                    ["colab", "drivemount", "-s", args.session, "/content/drive"],
                    timeout=15 * 60,
                    monitor_ram=False,
                )
                run_guarded(
                    ["colab", "ls", "-s", args.session, "/content/drive/MyDrive"],
                    timeout=2 * 60,
                )
            if args.hf_token_file is not None:
                token_path = args.hf_token_file.resolve()
                token = token_path.read_text(encoding="utf-8").strip()
                if not token.startswith("hf_") or any(char.isspace() for char in token):
                    raise GuardFailure("HF token file is invalid")
                run_guarded(
                    [
                        "colab", "upload", "-s", args.session,
                        str(token_path), "/content/.hf_token",
                    ],
                    timeout=5 * 60,
                )
                auth_script = temporary / "authenticate_hf.py"
                auth_script.write_text(
                    "from pathlib import Path\n"
                    "from huggingface_hub import login\n"
                    "path = Path('/content/.hf_token')\n"
                    "login(token=path.read_text(encoding='utf-8').strip(), "
                    "add_to_git_credential=False)\n"
                    "path.unlink()\n"
                    "print('Hugging Face authentication ready.')\n",
                    encoding="utf-8",
                )
                run_guarded(
                    ["colab", "exec", "-s", args.session, "-f", str(auth_script)],
                    timeout=5 * 60,
                )
            run_guarded(
                [
                    "colab", "upload", "-s", args.session,
                    str(archive), "/content/DeepRL_Monopoly.tar.gz",
                ],
                timeout=20 * 60,
            )
            restore_snapshot(args.session, temporary, artifact_dir)
            if args.rclone_config is not None:
                rclone_config, rclone_binary = validate_rclone_inputs(
                    args.rclone_config, args.rclone_binary
                )
                rclone_version, rclone_hash = rclone_reference(rclone_binary)
                run_guarded(
                    [
                        "colab", "upload", "-s", args.session,
                        str(rclone_config), "/content/rclone.conf",
                    ],
                    timeout=5 * 60,
                )
                run_guarded(
                    [
                        "colab", "upload", "-s", args.session,
                        str(DRIVE_SYNC), "/content/drive_checkpoint_sync.py",
                    ],
                    timeout=5 * 60,
                )
                start_sync = temporary / "start_drive_sync.py"
                start_sync.write_text(
                    "import hashlib, json, os, shutil, subprocess, sys, time, urllib.request, zipfile\n"
                    "from pathlib import Path\n"
                    f"version = {rclone_version!r}\n"
                    f"expected_hash = {rclone_hash!r}\n"
                    "archive = Path('/content/rclone.zip')\n"
                    "url = f'https://downloads.rclone.org/v{version}/rclone-v{version}-linux-amd64.zip'\n"
                    "with urllib.request.urlopen(url, timeout=180) as source, archive.open('wb') as target:\n"
                    "    shutil.copyfileobj(source, target)\n"
                    "temporary = Path('/content/rclone.tmp')\n"
                    "member = f'rclone-v{version}-linux-amd64/rclone'\n"
                    "with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, temporary.open('wb') as target:\n"
                    "    shutil.copyfileobj(source, target)\n"
                    "if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash:\n"
                    "    raise RuntimeError('Downloaded rclone binary hash mismatch')\n"
                    "os.replace(temporary, '/content/rclone')\n"
                    "archive.unlink()\n"
                    "os.chmod('/content/rclone', 0o700)\n"
                    "os.chmod('/content/rclone.conf', 0o600)\n"
                    "result = subprocess.run([\n"
                    "    '/content/rclone', '--config', '/content/rclone.conf',\n"
                    f"    'lsjson', {DRIVE_CHECKPOINT_ROOT!r} + '/rclone-smoke.txt', '--stat',\n"
                    "], check=True, capture_output=True, text=True)\n"
                    "item = json.loads(result.stdout)\n"
                    "if item.get('Name') != 'rclone-smoke.txt':\n"
                    "    raise RuntimeError('Drive checkpoint smoke marker is unavailable')\n"
                    "log = open('/content/drive_checkpoint_sync.log', 'ab', buffering=0)\n"
                    "process = subprocess.Popen("
                    "[sys.executable, '/content/drive_checkpoint_sync.py'], "
                    "stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, "
                    "start_new_session=True)\n"
                    "time.sleep(2)\n"
                    "if process.poll() is not None:\n"
                    "    raise RuntimeError('Drive checkpoint sync exited during startup')\n"
                    "print(f'Google Drive rclone checkpoint sync started: {process.pid}')\n",
                    encoding="utf-8",
                )
                run_guarded(
                    ["colab", "exec", "-s", args.session, "-f", str(start_sync)],
                    timeout=5 * 60,
                )
            if args.collection_progress is not None:
                if "collect" not in args.stages:
                    raise GuardFailure("Collection progress requires the collect stage")
                upload_collection_progress(
                    args.session, args.collection_progress.resolve(), temporary
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
                snapshot_needed = True
                run_guarded(
                    [
                        "colab", "exec", "-s", args.session,
                        "-f", str(stage_notebook), "--timeout", str(timeout_seconds),
                    ],
                    timeout=timeout_seconds + 5 * 60,
                    monitor_ram=False,
                )
                executed = stage_notebook.with_name(stage_notebook.stem + "_output.ipynb")
                if not executed.exists():
                    raise GuardFailure(f"Colab did not export the executed {stage} notebook")
                validate_executed_notebook(executed)
                if stage == "train" and args.rclone_config is not None:
                    wait_for_sync = temporary / "wait_for_drive_sync.py"
                    wait_for_sync.write_text(
                        "import json, time\n"
                        "from pathlib import Path\n"
                        f"complete = Path({REMOTE_RUN_ROOT!r}) / 'adapters/drive_sync_complete.json'\n"
                        "deadline = time.monotonic() + 1800\n"
                        "while time.monotonic() < deadline and not complete.exists():\n"
                        "    time.sleep(5)\n"
                        "if not complete.exists():\n"
                        "    raise RuntimeError('Drive checkpoint sync did not finish')\n"
                        "value = json.loads(complete.read_text(encoding='utf-8'))\n"
                        "if value.get('status') != 'complete':\n"
                        "    raise RuntimeError('Drive checkpoint sync is incomplete')\n"
                        "print('Google Drive training artifacts verified uploaded.')\n",
                        encoding="utf-8",
                    )
                    run_guarded(
                        [
                            "colab", "exec", "-s", args.session,
                            "-f", str(wait_for_sync), "--timeout", "1800",
                        ],
                        timeout=31 * 60,
                        monitor_ram=False,
                    )
                local_executed = artifact_dir / executed.name
                shutil.copy2(executed, local_executed)
                log_path = artifact_dir / f"{stage}_execution.md"
                run_guarded(
                    ["colab", "log", "-s", args.session, "-o", str(log_path)],
                    timeout=10 * 60,
                )
                snapshot = snapshot_run(args.session, temporary, artifact_dir, stage)
                snapshot_needed = False
                print(f"Downloaded resumable {stage} snapshot to {snapshot}", flush=True)
        finally:
            if cleanup_needed:
                if snapshot_needed:
                    try:
                        snapshot_run(
                            args.session,
                            temporary,
                            artifact_dir,
                            "shutdown",
                            min_ram_gib=0.5,
                        )
                    except Exception as exc:
                        print(f"WARNING: final pilot snapshot failed: {exc}", flush=True)
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
