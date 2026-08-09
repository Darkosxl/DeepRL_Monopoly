"""Command line entry points for the frozen benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .config import BenchmarkConfig, SearchConfig
from .engine import DICE_OUTCOMES, ROOT, ActionType, SharedGame, transition
from .exports import export_teacher
from .ladder import evaluate_promotion, gate_run_directory
from .model import MonopolyZeroNet
from .search import MaxNPUCT
from .training import Trainer, collect_asu_examples, save_asu_examples


DEFAULT_PPO = ROOT / "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"


def smoke() -> dict:
    before = random.getstate()
    game = SharedGame.new(17, max_rounds=2)
    if random.getstate() != before:
        raise RuntimeError("Game construction leaked RNG state")
    actor = game.env.whose_turn()
    game.step(int(ActionType.END_TURN))
    roll_state = game.env
    for outcome in DICE_OUTCOMES:
        branch = transition(roll_state, int(ActionType.ROLL_DICE), outcome)
        if branch.last_dice != outcome:
            raise RuntimeError(f"Chance branch mismatch: {outcome}")
    if random.getstate() != before:
        raise RuntimeError("Explicit dice branches leaked RNG state")

    model = MonopolyZeroNet()
    model.load_ppo_actor(DEFAULT_PPO)
    game = SharedGame.new(23, max_rounds=2)
    actor = game.env.whose_turn()
    property_ = game.env.properties[1]
    property_.owner = actor
    game.env.players[actor].properties.append(property_)
    result = MaxNPUCT(
        model,
        SearchConfig(simulations=4, max_depth=16),
    ).choose_action(game, actor, 101)
    if result.chosen_action not in game.env.get_allowed_actions(actor):
        raise RuntimeError("Smoke search selected an illegal action")
    return {
        "status": "ok",
        "ruleset": "ppo-plus-v2",
        "dice_outcomes": len(DICE_OUTCOMES),
        "simulations": result.simulations,
        "chosen_action": result.chosen_action,
        "latency_s": result.latency_s,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m monopoly_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")

    train = subparsers.add_parser("train")
    train.add_argument("--run-dir", required=True)
    train.add_argument("--bootstrap-ppo", default=str(DEFAULT_PPO))
    train.add_argument("--generations", type=int, default=1)
    train.add_argument("--device", default="auto")
    train.add_argument("--fallback-colab", action="store_true")
    train.add_argument("--asu-expert-data", nargs="*", default=())

    collect_asu = subparsers.add_parser("collect-asu")
    collect_asu.add_argument("--output", required=True)
    collect_asu.add_argument("--games", type=int, required=True)
    collect_asu.add_argument("--seed-base", type=int, required=True)
    collect_asu.add_argument("--max-rounds", type=int, default=200)
    collect_asu.add_argument("--rollout-positions", type=int, default=0)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--run-dir", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--champion", required=True)
    evaluate.add_argument("--candidate", required=True)

    teacher = subparsers.add_parser("export-teacher")
    teacher.add_argument("--champion", required=True)
    teacher.add_argument("--games", type=int, default=256)
    teacher.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        reports = Trainer(
            args.run_dir,
            args.bootstrap_ppo,
            device=args.device,
            fallback_colab=args.fallback_colab,
            asu_expert_data=args.asu_expert_data,
        ).run(args.generations)
        print(json.dumps(reports, indent=2, sort_keys=True))
        return 0
    if args.command == "collect-asu":
        examples = collect_asu_examples(
            games=args.games,
            seed_base=args.seed_base,
            max_rounds=args.max_rounds,
            rollout_positions=args.rollout_positions,
        )
        destination = save_asu_examples(args.output, examples)
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "positions": len(examples["states"]),
                    "rollout_positions": int(examples["teachers"].sum()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "gate":
        report, release = gate_run_directory(args.run_dir)
        print(json.dumps({"report": report, "release": None if release is None else str(release)}, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    if args.command == "evaluate":
        config = BenchmarkConfig()
        summary, passed = evaluate_promotion(
            args.candidate,
            args.champion,
            config=config,
            seed_base=config.seeds.ladder,
        )
        print(json.dumps({"candidate": summary.as_dict(), "promotion_passed": passed}, indent=2, sort_keys=True))
        return 0

    output = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "teacher-data" / Path(args.champion).stem
    manifest = export_teacher(args.champion, games=args.games, output_dir=output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
