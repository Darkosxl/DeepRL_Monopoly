"""
train_and_save.py
-----------------
Trains a hybrid PPO or DDQN agent against the three fixed-policy opponents
and saves a resumable model checkpoint.

Usage:
    python tools/train_and_save.py                        # default 2000 games
    python tools/train_and_save.py --games 5000           # more training
    python tools/train_and_save.py --algo ddqn --games 10000
    python tools/train_and_save.py --algo ppo --games 2000 --out my_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_guard import GIB, MemoryWatchdog
from monopoly_game_engine import train_ddqn, train_ppo


def _history_segment(history: dict) -> dict:
    return {
        key: history.get(key)
        for key in (
            "resumed_from_games",
            "games_completed_this_run",
            "games_completed",
            "elapsed_seconds",
            "peak_rss_gib",
            "peak_cuda_gib",
        )
    }


def merge_training_history(previous: dict | None, current: dict) -> dict:
    """Merge a completed resume segment into its checkpoint history."""
    if not previous:
        current["training_segments"] = [_history_segment(current)]
        return current

    expected = int(current.get("resumed_from_games", 0))
    actual = int(previous.get("games_completed", 0))
    if actual != expected:
        raise ValueError(
            f"History/checkpoint mismatch: history has {actual} games, "
            f"checkpoint resumed from {expected}."
        )

    merged = dict(current)
    series_keys = {
        key
        for history in (previous, current)
        for key, value in history.items()
        if isinstance(value, list) and key != "training_segments"
    }
    for key in series_keys:
        merged[key] = list(previous.get(key, [])) + list(current.get(key, []))

    previous_segments = previous.get("training_segments")
    if previous_segments is None:
        previous_segments = [_history_segment(previous)]
    merged["training_segments"] = list(previous_segments) + [
        _history_segment(current)
    ]
    merged["resumed_from_games"] = int(previous.get("resumed_from_games", 0))
    merged["games_completed_this_run"] = int(
        previous.get("games_completed_this_run", actual)
    ) + int(current.get("games_completed_this_run", 0))
    merged["elapsed_seconds"] = float(previous.get("elapsed_seconds", 0.0)) + float(
        current.get("elapsed_seconds", 0.0)
    )
    if merged["games_completed_this_run"]:
        merged["seconds_per_game"] = (
            merged["elapsed_seconds"] / merged["games_completed_this_run"]
        )
    for key in ("peak_rss_gib", "peak_cuda_gib"):
        merged[key] = max(
            float(previous.get(key, 0.0)), float(current.get(key, 0.0))
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train and save a Monopoly DRL agent")
    parser.add_argument(
        "--algo",
        choices=["ppo", "ddqn"],
        default="ppo",
        help="Algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Use hybrid mode (default: True)",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="hybrid",
        action="store_false",
        help="Disable hybrid mode (use standard DRL)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2000,
        help="Number of training games (default: 2000)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output path for saved model weights"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device (default: auto)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-rss-gib", type=float, default=3)
    parser.add_argument("--hard-rss-gib", type=float, default=4)
    parser.add_argument("--min-available-gib", type=float, default=2)
    args = parser.parse_args()

    # Default output filename
    if args.out is None:
        mode = "hybrid" if args.hybrid else "standard"
        args.out = str(ROOT / "artifacts" / "ppo_plus" / f"{args.algo}_{mode}_model.pt")
    if args.resume and not Path(args.out).exists():
        parser.error(f"resume checkpoint does not exist: {args.out}")
    history_path = str(Path(args.out).with_suffix("")) + "_history.json"
    previous_history = None
    if args.resume and Path(history_path).exists():
        with open(history_path) as f:
            previous_history = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Mode      : {'Hybrid' if args.hybrid else 'Standard'}")
    print(f"  Games     : {args.games}")
    print(f"  Device    : {args.device}")
    print(f"  Save to   : {args.out}")
    print(f"{'=' * 60}\n")

    start = time.time()
    watchdog = MemoryWatchdog(
        stop_rss_gib=args.stop_rss_gib,
        hard_rss_gib=args.hard_rss_gib,
        min_available_gib=args.min_available_gib,
    )

    if args.algo == "ppo":
        agent, history = train_ppo(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=max(1, args.games // 50),
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=args.out,
            watchdog=watchdog,
            seed=args.seed,
            resume_path=args.out if args.resume else None,
        )
    else:
        agent, history = train_ddqn(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=max(1, args.games // 50),
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=args.out,
            watchdog=watchdog,
            seed=args.seed,
            resume_path=args.out if args.resume else None,
        )

    elapsed = time.time() - start
    games_completed = history.get("games_completed", 0)
    games_this_run = history.get("games_completed_this_run", games_completed)
    peak_rss_gib = watchdog.peak_rss / GIB
    peak_cuda_gib = 0.0
    status = "stopped early" if history.get("stopped_early") else "complete"
    print(f"\nTraining {status} in {elapsed:.1f}s")
    print(f"Games completed: {games_completed}/{args.games}")
    if games_this_run:
        print(f"Mean wall time: {elapsed / games_this_run:.3f}s/game")
    print(f"Peak process RSS: {peak_rss_gib:.2f} GiB")
    if getattr(agent, "device", torch.device("cpu")).type == "cuda":
        peak_cuda_gib = torch.cuda.max_memory_allocated() / GIB
        print(f"Peak CUDA memory: {peak_cuda_gib:.2f} GiB")

    history["elapsed_seconds"] = elapsed
    history["seconds_per_game"] = elapsed / games_this_run if games_this_run else None
    history["peak_rss_gib"] = peak_rss_gib
    history["peak_cuda_gib"] = peak_cuda_gib

    # Save model weights
    agent.save(args.out)
    print(f"Model saved to: {args.out}")

    # Save training history
    history = merge_training_history(previous_history, history)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Print final win rate
    if history.get("win_rates"):
        final_wr = history["win_rates"][-1]
        best_wr = max(history["win_rates"])
        print(f"\nFinal win rate : {final_wr:.1f}%")
        print(f"Best win rate  : {best_wr:.1f}%")


if __name__ == "__main__":
    main()
