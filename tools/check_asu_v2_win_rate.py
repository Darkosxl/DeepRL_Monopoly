"""
check_asu_v2_win_rate.py
-------------------------
Scriptable pass/fail win-rate gate for an ASU v2 policy against a fixed
lineup of opponents, using ASU_FROZEN_TEACHER.evaluate.evaluate_lineup's
seat-balanced seed blocks. Requires BOTH the raw win rate and the 95%
Wilson lower bound to clear their thresholds -- matching this repo's own
convention of pairing a raw-rate threshold with a Wilson-lower threshold
roughly 10 points below it (see GateConfig/TrainingConfig in
monopoly_bench/config.py). Exits non-zero on failure so it can gate a
pipeline instead of requiring someone to read a number out of a JSON blob.

Usage examples
--------------
  # Pilot: fast sanity signal (32 games)
  python tools/check_asu_v2_win_rate.py \\
    --focus asu-rollout-v2 --opponents asu-rollout-v1 asu-rollout-v1 asu-rollout-v1 \\
    --seeds-start 9900000 --seeds-count 8 \\
    --target-win-rate 0.70 --target-wilson-lower 0.60

  # Confirmatory run (120 games)
  python tools/check_asu_v2_win_rate.py \\
    --focus asu-rollout-v2 --opponents asu-rollout-v1 asu-rollout-v1 asu-rollout-v1 \\
    --seeds-start 9900000 --seeds-count 30 \\
    --target-win-rate 0.70 --target-wilson-lower 0.60 \\
    --output artifacts/asu_v2_vs_3xrollout_v1.json

  # Secondary/easier matchup
  python tools/check_asu_v2_win_rate.py \\
    --focus asu-rollout-v2 --opponents asu-value-v1 asu-value-v1 asu-value-v1 \\
    --seeds-start 9950000 --seeds-count 30 --target-win-rate 0.70 --target-wilson-lower 0.60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pass/fail win-rate gate for an ASU v2 policy")
    parser.add_argument("--focus", required=True)
    parser.add_argument("--opponents", nargs=3, required=True, metavar=("A", "B", "C"))
    parser.add_argument("--seeds-start", type=int, default=9_900_000)
    parser.add_argument("--seeds-count", type=int, default=30)
    parser.add_argument("--target-win-rate", type=float, default=0.70)
    parser.add_argument("--target-wilson-lower", type=float, default=0.60)
    parser.add_argument("--max-decisions", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(range(args.seeds_start, args.seeds_start + args.seeds_count))
    games = args.seeds_count * 4
    print(
        f"Evaluating {args.focus} vs {args.opponents} over {len(seeds)} seeds "
        f"({games} games, seat-balanced)..."
    )
    started = time.perf_counter()
    result = evaluate_lineup(args.focus, tuple(args.opponents), seeds=seeds, max_decisions=args.max_decisions)
    elapsed = time.perf_counter() - started

    summary = result["win_rates"].get(args.focus)
    if summary is None or summary["win_rate"] is None:
        print(f"FAIL: no completed games recorded for {args.focus!r}")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    win_rate = summary["win_rate"]
    wilson_lower, wilson_upper = summary["wilson_95"]
    passed = win_rate >= args.target_win_rate and wilson_lower > args.target_wilson_lower

    print(f"Completed in {elapsed:.1f}s ({result['truncations']} truncated games)")
    print(f"win_rate={win_rate:.4f}  wilson_95=({wilson_lower:.4f}, {wilson_upper:.4f})")
    print(
        f"target: win_rate>={args.target_win_rate}  wilson_lower>{args.target_wilson_lower}"
    )
    print("PASS" if passed else "FAIL")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        payload["gate"] = {
            "focus": args.focus,
            "opponents": list(args.opponents),
            "target_win_rate": args.target_win_rate,
            "target_wilson_lower": args.target_wilson_lower,
            "observed_win_rate": win_rate,
            "observed_wilson_95": [wilson_lower, wilson_upper],
            "passed": passed,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
