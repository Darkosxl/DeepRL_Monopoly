"""
benchmark_asu_v2.py
--------------------
Wall-clock speed comparison between the frozen ASU v1 teachers
(ASUValueV1/ASURolloutV1) and the new v2 variants (ASUValueV2/ASURolloutV2)
across a few representative game states.

ASURolloutV1 is extremely slow (minutes per decision in a developed game),
so the rollout comparison defaults to a single repeat per state and prints
progress as it runs. Use --skip-rollout to only time the cheaper
ASUValueV1 vs ASUValueV2 comparison.

Usage examples
--------------
  python tools/benchmark_asu_v2.py
  python tools/benchmark_asu_v2.py --value-repeats 10 --skip-rollout
  python tools/benchmark_asu_v2.py --output artifacts/asu_v2_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Callable

from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1
from ASU_FROZEN_TEACHER.core_v2 import ASURolloutV2, ASUValueV2
from monopoly_game_engine.constants import PROPERTY_IDS
from monopoly_game_engine.env import PHASE_POST_ROLL, MonopolyEnv


def _new_env() -> MonopolyEnv:
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.turn_order = [0, 1, 2, 3]
    env.current_turn_idx = 0
    return env


def _fresh_state() -> MonopolyEnv:
    env = _new_env()
    env.phase = PHASE_POST_ROLL
    env.has_rolled = True
    env.players[0].position = 1
    return env


def _light_state() -> MonopolyEnv:
    env = _new_env()
    for square in PROPERTY_IDS[0:2]:
        env.properties[square].owner = 0
        env.players[0].properties.append(env.properties[square])
    for square in PROPERTY_IDS[2:4]:
        env.properties[square].owner = 1
        env.players[1].properties.append(env.properties[square])
    env._update_monopolies()
    for player in env.players:
        player.cash = 1500
    env.phase = PHASE_POST_ROLL
    env.has_rolled = True
    env.players[0].position = 6
    return env


def _heavy_trade_state() -> MonopolyEnv:
    env = _new_env()
    for square in PROPERTY_IDS[0:3]:
        env.properties[square].owner = 0
        env.players[0].properties.append(env.properties[square])
    for square in PROPERTY_IDS[3:9]:
        env.properties[square].owner = 1
        env.players[1].properties.append(env.properties[square])
    env._update_monopolies()
    for player in env.players:
        player.cash = 1500
    return env


STATES: dict[str, Callable[[], MonopolyEnv]] = {
    "fresh": _fresh_state,
    "light": _light_state,
    "heavy_trade": _heavy_trade_state,
}


def _time_calls(build_env: Callable[[], MonopolyEnv], decide: Callable[[MonopolyEnv], object], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        env = build_env()
        started = time.perf_counter()
        decide(env)
        samples.append(time.perf_counter() - started)
    return samples


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def _report_pair(label: str, v1_samples: list[float], v2_samples: list[float]) -> dict:
    v1_summary = _summarize(v1_samples)
    v2_summary = _summarize(v2_samples)
    speedup = v1_summary["mean"] / v2_summary["mean"] if v2_summary["mean"] > 0 else float("inf")
    print(
        f"  {label}: v1 mean={v1_summary['mean']:.4f}s  v2 mean={v2_summary['mean']:.4f}s  "
        f"speedup={speedup:.2f}x"
    )
    return {"v1": v1_summary, "v2": v2_summary, "speedup": speedup}


def run(value_repeats: int, rollout_repeats: int, skip_rollout: bool, seed: int) -> dict:
    random.seed(seed)
    results: dict[str, dict] = {"value": {}, "rollout": {}}

    print(f"ASUValueV1 vs ASUValueV2 ({value_repeats} repeats per state)")
    for name, build in STATES.items():
        v1_samples = _time_calls(build, lambda env: ASUValueV1(0).decide(env), value_repeats)
        v2_samples = _time_calls(build, lambda env: ASUValueV2(0).decide(env), value_repeats)
        results["value"][name] = _report_pair(name, v1_samples, v2_samples)

    if skip_rollout:
        print("Skipping ASURolloutV1 vs ASURolloutV2 (--skip-rollout)")
        return results

    print(
        f"\nASURolloutV1 vs ASURolloutV2 ({rollout_repeats} repeat(s) per state; "
        "ASURolloutV1 can take minutes per call)"
    )
    for name, build in STATES.items():
        print(f"  {name}: timing ASURolloutV1...", flush=True)
        v1_samples = _time_calls(build, lambda env: ASURolloutV1(0).decide(env), rollout_repeats)
        print(f"  {name}: timing ASURolloutV2...", flush=True)
        v2_samples = _time_calls(build, lambda env: ASURolloutV2(0).decide(env), rollout_repeats)
        results["rollout"][name] = _report_pair(name, v1_samples, v2_samples)

    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time ASU v1 vs v2 decide() calls")
    parser.add_argument("--value-repeats", type=int, default=5)
    parser.add_argument("--rollout-repeats", type=int, default=1)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = run(args.value_repeats, args.rollout_repeats, args.skip_rollout, args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
