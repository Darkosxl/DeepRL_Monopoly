from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path


RUN_ROOT = Path("/content/asu_pilot_v1")
ADAPTER_DIR = RUN_ROOT / "adapters"
CHECKPOINT_DIR = ADAPTER_DIR / "checkpoints"
RCLONE = Path("/content/rclone")
RCLONE_CONFIG = Path("/content/rclone.conf")
STATE_PATH = ADAPTER_DIR / "drive_sync_state.json"
COMPLETE_PATH = ADAPTER_DIR / "drive_sync_complete.json"
DRIVE_ROOT = (
    "gemma_drive:DeepRL_Monopoly/Gemma4_12B_Monopoly/"
    "pilot_v1/adapters/checkpoints"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def upload(source: Path, name: str | None = None) -> None:
    subprocess.run(
        [
            str(RCLONE),
            "--config",
            str(RCLONE_CONFIG),
            "copyto",
            str(source),
            f"{DRIVE_ROOT}/{name or source.name}",
            "--checksum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def checkpoint_ready(path: Path) -> bool:
    required = (path / "trainer_state.json", path / "adapter_model.safetensors")
    if not all(item.is_file() for item in required):
        return False
    before = sorted((item.name, item.stat().st_size) for item in path.iterdir())
    time.sleep(10)
    after = sorted((item.name, item.stat().st_size) for item in path.iterdir())
    return before == after


def archive_checkpoint(path: Path) -> Path:
    destination = Path("/content") / f"{path.name}.tar.gz"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(path, arcname=path.name)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    if not RCLONE.is_file() or not RCLONE_CONFIG.is_file():
        raise RuntimeError("rclone binary/config is unavailable")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    state = {"uploaded_steps": []}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    uploaded = {int(step) for step in state.get("uploaded_steps", [])}
    deadline = time.monotonic() + 12 * 3600

    while time.monotonic() < deadline:
        try:
            checkpoints = sorted(
                (path for path in CHECKPOINT_DIR.glob("checkpoint-*") if path.is_dir()),
                key=checkpoint_step,
            )
            for checkpoint in checkpoints:
                step = checkpoint_step(checkpoint)
                if step in uploaded or not checkpoint_ready(checkpoint):
                    continue
                archive = archive_checkpoint(checkpoint)
                manifest_path = Path("/content") / f"{checkpoint.name}.json"
                fingerprint_path = CHECKPOINT_DIR / "checkpoint_fingerprint.json"
                manifest = {
                    "step": step,
                    "archive": archive.name,
                    "sha256": sha256(archive),
                    "fingerprint": (
                        json.loads(fingerprint_path.read_text(encoding="utf-8")).get("fingerprint")
                        if fingerprint_path.exists() else None
                    ),
                }
                atomic_json(manifest_path, manifest)
                upload(archive)
                upload(manifest_path)
                uploaded.add(step)
                state = {"uploaded_steps": sorted(uploaded), "latest": manifest}
                atomic_json(STATE_PATH, state)
                print(f"Uploaded Drive checkpoint {step}", flush=True)

            run_manifest = RUN_ROOT / "manifest.json"
            train_complete = False
            if run_manifest.exists():
                value = json.loads(run_manifest.read_text(encoding="utf-8"))
                train_complete = value.get("stages", {}).get("train", {}).get("status") == "complete"
            if train_complete:
                final_artifacts = {}
                for artifact in (
                    ADAPTER_DIR / "gemma4_12b_monopoly_lora.zip",
                    ADAPTER_DIR / "sample_generations.json",
                    ADAPTER_DIR / "training_metrics.json",
                ):
                    if not artifact.is_file():
                        raise RuntimeError(f"Missing final training artifact: {artifact.name}")
                    upload(artifact)
                    final_artifacts[artifact.name] = sha256(artifact)
                final = {
                    "status": "complete",
                    "uploaded_steps": sorted(uploaded),
                    "artifacts": final_artifacts,
                }
                atomic_json(COMPLETE_PATH, final)
                upload(COMPLETE_PATH)
                RCLONE_CONFIG.unlink(missing_ok=True)
                print(json.dumps(final, sort_keys=True), flush=True)
                return
        except Exception as exc:
            print(f"Drive sync retry: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(15)
    raise RuntimeError("Drive checkpoint sync deadline reached")


if __name__ == "__main__":
    main()
