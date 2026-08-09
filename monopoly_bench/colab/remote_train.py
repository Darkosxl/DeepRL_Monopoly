"""Executed inside the named Colab session by monopoly_bench.colab."""

from pathlib import Path
import shutil
import subprocess
import tarfile

with tarfile.open("/content/DeepRL_Monopoly.tar.gz", "r:gz") as archive:
    archive.extractall("/content", filter="data")
with tarfile.open("/content/monopolyzero-baselines.tar.gz", "r:gz") as archive:
    archive.extractall("/content/DeepRL_Monopoly", filter="data")
run_dir = Path("/content/monopolyzero-ppo-plus-v2-v1")
run_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/monopolyzero-resume.tar.gz", "r:gz") as archive:
    archive.extractall("/content", filter="data")
restored = Path("/content/run")
if restored.exists():
    shutil.copytree(restored, run_dir, dirs_exist_ok=True)
try:
    subprocess.run(
        [
            "python",
            "-m",
            "monopoly_bench",
            "train",
            "--run-dir",
            str(run_dir),
            "--bootstrap-ppo",
            "/content/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt",
        ],
        cwd="/content/DeepRL_Monopoly",
        check=True,
    )
finally:
    with tarfile.open("/content/monopolyzero-result.tar.gz", "w:gz") as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.add(
                    path,
                    arcname=str(Path("run") / path.relative_to(run_dir)),
                )
