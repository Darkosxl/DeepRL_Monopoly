"""Executed inside the named Colab session by monopoly_bench.colab."""

from pathlib import Path
import shutil
import subprocess
import tarfile

from google.colab import drive


drive.mount("/content/drive")
with tarfile.open("/content/DeepRL_Monopoly.tar.gz", "r:gz") as archive:
    archive.extractall("/content", filter="data")
with tarfile.open("/content/monopolyzero-baselines.tar.gz", "r:gz") as archive:
    archive.extractall("/content/DeepRL_Monopoly", filter="data")
drive_run = Path("/content/drive/MyDrive/monopolyzero-ppo-plus-v2-v1")
drive_run.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/monopolyzero-resume.tar.gz", "r:gz") as archive:
    archive.extractall(drive_run.parent, filter="data")
restored = drive_run.parent / "run"
if restored.exists():
    shutil.copytree(restored, drive_run, dirs_exist_ok=True)
subprocess.run(
    [
        "python",
        "-m",
        "monopoly_bench",
        "train",
        "--run-dir",
        str(drive_run),
        "--bootstrap-ppo",
        "/content/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt",
    ],
    cwd="/content/DeepRL_Monopoly",
    check=True,
)
with tarfile.open("/content/monopolyzero-result.tar.gz", "w:gz") as archive:
    for path in drive_run.rglob("*"):
        if path.is_file():
            archive.add(path, arcname=str(Path("run") / path.relative_to(drive_run)))
