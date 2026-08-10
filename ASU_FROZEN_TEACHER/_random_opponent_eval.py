"""Ad-hoc seat-balanced eval of asu-value-v1 vs uniformly-random opponents.

Not part of the public CLI: ``evaluate.py`` only recognizes ``fixed-a``..``fixed-f``,
``asu-value-v1``/``asu-rollout-v1``, and ``ppo``/``ddqn``/``cfr`` checkpoint specs.
This reuses its private ``_run_game`` and ``wilson_interval`` helpers with a
minimal random-action adapter standing in for a fourth agent kind.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ASU_FROZEN_TEACHER.core import RULESET_VERSION, preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import (
    AgentFactory,
    AgentSpec,
    RESULTS_DISCLAIMER,
    _run_game,
    wilson_interval,
)
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH
from monopoly_game_engine.constants import NUM_PLAYERS

DEFAULT_MAX_DECISIONS = 20_000
FOCUS = AgentSpec("asu-value-v1", "asu-value-v1")
OPPONENT = AgentSpec("random", "random")


class RandomAdapter:
    """Uniformly samples any legal action; deterministic per (seed, seat)."""

    def __init__(self, player_id: int, seed: int):
        self.player_id = player_id
        self._rng = random.Random(seed)
        self.fallbacks = 0

    def choose_action(self, env) -> int:
        allowed = list(env.get_allowed_actions(self.player_id))
        return self._rng.choice(allowed)


class RandomCapableFactory(AgentFactory):
    def __init__(self) -> None:
        super().__init__()
        self.game_seed = 0

    def build(self, spec: AgentSpec, player_id: int):
        if spec.kind == "random":
            return RandomAdapter(player_id, seed=self.game_seed * 97 + player_id)
        return super().build(spec, player_id)


def evaluate_vs_random(seeds: tuple[int, ...], max_decisions: int = DEFAULT_MAX_DECISIONS) -> dict:
    factory = RandomCapableFactory()
    games = []
    started = time.perf_counter()
    with preserve_global_rng():
        for seed in seeds:
            factory.game_seed = seed
            for focus_seat in range(NUM_PLAYERS):
                specs = [OPPONENT] * NUM_PLAYERS
                specs[focus_seat] = FOCUS
                games.append(
                    _run_game(tuple(specs), focus_seat, int(seed), max_decisions, factory)
                )

    identifiers = {"asu-value-v1", "random"}
    summaries = {}
    net_worth = {}
    for identifier in sorted(identifiers):
        appearances = [
            (game, seat)
            for game in games
            if not game["truncated"]
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        wins = sum(game["winner"] == seat for game, seat in appearances)
        interval = wilson_interval(wins, len(appearances))
        all_appearances = [
            (game, seat)
            for game in games
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        worth = [game["final_net_worth"][seat] for game, seat in all_appearances]
        rate = wins / len(appearances) if appearances else None
        summaries[identifier] = {
            "wins": wins,
            "games": len(appearances),
            "truncated_appearances": len(all_appearances) - len(appearances),
            "win_rate": rate,
            "win_rate_percent": None if rate is None else 100 * rate,
            "wilson_95": list(interval),
        }
        net_worth[identifier] = {
            "mean": sum(worth) / len(worth) if worth else None,
            "values": worth,
        }

    return {
        "ruleset": RULESET_VERSION,
        "policy_ids": {"focus": "asu-value-v1", "opponents": ["random", "random", "random"]},
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "checkpoint_hashes": {},
        "seeds": list(seeds),
        "paired_block_size": NUM_PLAYERS,
        "games": games,
        "win_rates": summaries,
        "final_net_worth": net_worth,
        "truncations": sum(game["truncated"] for game in games),
        "scripted_compatibility_fallbacks": sum(
            sum(game["scripted_compatibility_fallbacks"]) for game in games
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "disclaimer": RESULTS_DISCLAIMER,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_vs_random(tuple(args.seeds), args.max_decisions)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
