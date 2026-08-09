"""
train_and_save.py
-----------------
Trains a Hybrid PPO agent (the best performing model from the paper)
against the three fixed-policy opponents and saves the model weights.

Usage:
    python train_and_save.py                        # default 2000 games
    python train_and_save.py --games 5000           # more training
    python train_and_save.py --algo ddqn --games 10000
    python train_and_save.py --algo ppo --games 2000 --out my_model.pt
"""

import argparse
import json
import time
from pathlib import Path

from monopoly_drl import train_ddqn, train_ppo


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
    args = parser.parse_args()

    # Default output filename
    if args.out is None:
        mode = "hybrid" if args.hybrid else "standard"
        root = Path(__file__).resolve().parents[1]
        args.out = str(root / "artifacts" / "ppo_plus" / f"{args.algo}_{mode}_model.pt")

    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Mode      : {'Hybrid' if args.hybrid else 'Standard'}")
    print(f"  Games     : {args.games}")
    print(f"  Device    : {args.device}")
    print(f"  Save to   : {args.out}")
    print(f"{'=' * 60}\n")

    start = time.time()

    if args.algo == "ppo":
        agent, history = train_ppo(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=max(1, args.games // 50),
            device=args.device,
        )
    else:
        agent, history = train_ddqn(
            hybrid=args.hybrid,
            player_id=0,
            n_games=args.games,
            log_every=max(1, args.games // 50),
        )

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save model weights
    agent.save(args.out)
    print(f"Model saved to: {args.out}")

    # Save training history
    history_path = str(Path(args.out).with_suffix("")) + "_history.json"
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
