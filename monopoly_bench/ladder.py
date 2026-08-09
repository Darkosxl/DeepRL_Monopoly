"""Seat-balanced promotion, extreme king gate, and immutable freeze bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from ASU_FROZEN_TEACHER import FROZEN_SPEC_HASH

from .adapters import ASUAdapter, SearchAdapter, available_baselines
from .arena import balanced_single_seats, balanced_team_seats, play_game, summarize
from .config import BenchmarkConfig
from .contracts import MatchSummary
from .engine import engine_hashes, source_hashes
from .model import MonopolyZeroNet
from .storage import _atomic_json


BENCH_ROOT = Path(__file__).resolve().parent
RELEASE_TAG = "monopolyzero-ppo-plus-v2-v1"


def _safe(summary: MatchSummary) -> bool:
    return (
        summary.completed == summary.games
        and summary.illegal_actions == 0
        and summary.fallbacks == 0
        and summary.crashes == 0
    )


def promotion_passes(summary: MatchSummary, config: BenchmarkConfig) -> bool:
    return (
        _safe(summary)
        and summary.win_rate >= config.training.promotion_win_rate
        and summary.wilson_lower > config.training.promotion_wilson_lower
    )


def evaluate_promotion(
    challenger: str | Path,
    incumbent: str | Path,
    *,
    config: BenchmarkConfig,
    seed_base: int | None = None,
) -> tuple[MatchSummary, bool]:
    challenger_model = MonopolyZeroNet.load_inference(challenger)
    incumbent_model = MonopolyZeroNet.load_inference(incumbent)
    seats = balanced_team_seats(config.training.promotion_games)
    results = []
    start = config.seeds.promotion if seed_base is None else seed_base
    for index, target in enumerate(seats):
        policies = {
            seat: SearchAdapter(
                challenger_model if seat in target else incumbent_model,
                config.search,
                self_play=False,
            )
            for seat in range(4)
        }
        results.append(
            play_game(
                game_id=index,
                seed=start + index,
                policies=policies,
                max_rounds=config.max_rounds,
                record_seats=set(),
            )
        )
    summary = summarize(results, seats)
    passed = promotion_passes(summary, config)
    return summary, passed


def evaluate_baseline(
    champion: str | Path,
    baseline: object | tuple[object, object, object],
    *,
    games: int,
    config: BenchmarkConfig,
    seed_base: int,
) -> MatchSummary:
    model = MonopolyZeroNet.load_inference(champion)
    opponents = baseline if isinstance(baseline, tuple) else (baseline,) * 3
    if len(opponents) != 3:
        raise ValueError("A four-player baseline matchup needs three opponents")
    seats = balanced_single_seats(games)
    results = []
    for index, target in enumerate(seats):
        champion_seat = next(iter(target))
        policies = {champion_seat: SearchAdapter(model, config.search, self_play=False)}
        policies.update(
            zip(
                (seat for seat in range(4) if seat != champion_seat),
                opponents,
            )
        )
        results.append(
            play_game(
                game_id=index,
                seed=seed_base + index,
                policies=policies,
                max_rounds=config.max_rounds,
                record_seats=set(),
            )
        )
    return summarize(results, seats)


def full_gate_passes(summary: MatchSummary, config: BenchmarkConfig) -> bool:
    return (
        _safe(summary)
        and summary.win_rate >= config.gate.min_win_rate
        and summary.wilson_lower > config.gate.min_wilson_lower
        and summary.latency_p95_s <= config.gate.max_latency_p95_s
    )


def run_gate(
    champion: str | Path,
    *,
    config: BenchmarkConfig,
    ppo_checkpoint: str | Path | None = None,
    cfr_checkpoint: str | Path | None = None,
) -> dict:
    config.assert_frozen_v1()
    champion = Path(champion)
    MonopolyZeroNet.load_inference(champion)
    baselines = available_baselines(ppo_checkpoint, cfr_checkpoint)
    matchups = dict(baselines)
    mixed_names = ("asu_value_v1", "TheDealMaker", "TheGambler")
    if all(name in baselines for name in mixed_names):
        matchups["mixed_asu_dealmaker_gambler"] = tuple(
            baselines[name] for name in mixed_names
        )
    screens = {}
    offset = 0
    for name, baseline in matchups.items():
        summary = evaluate_baseline(
            champion,
            baseline,
            games=config.gate.screen_games,
            config=config,
            seed_base=config.seeds.gate + offset,
        )
        screens[name] = summary.as_dict()
        offset += config.gate.screen_games
    screen_passed = all(
        value["completed"] == value["games"]
        and value["illegal_actions"] == value["fallbacks"] == value["crashes"] == 0
        and value["win_rate"] >= config.gate.screen_win_rate
        for value in screens.values()
    )

    rollout_required = "asu_value_v1" in baselines
    rollout_screen = None
    rollout_screen_passed = not rollout_required
    if screen_passed and rollout_required:
        summary = evaluate_baseline(
            champion,
            ASUAdapter(rollout=True),
            games=config.gate.screen_games,
            config=config,
            seed_base=config.seeds.gate + 50_000,
        )
        rollout_screen = summary.as_dict()
        rollout_screen_passed = (
            _safe(summary) and summary.win_rate >= config.gate.screen_win_rate
        )

    full = {}
    if screen_passed and rollout_screen_passed:
        full_start = config.seeds.gate + 100_000
        for index, (name, baseline) in enumerate(matchups.items()):
            summary = evaluate_baseline(
                champion,
                baseline,
                games=config.gate.full_games,
                config=config,
                seed_base=full_start + index * config.gate.full_games,
            )
            full[name] = summary.as_dict()
    passed = screen_passed and rollout_screen_passed and bool(full) and all(
        full_gate_passes(MatchSummary(**value), config) for value in full.values()
    )
    return {
        "schema": 1,
        "ruleset": config.ruleset,
        "scope": "repository ppo-plus-v2 benchmark; not official or professional Monopoly",
        "candidate": str(champion.resolve()),
        "candidate_sha256": _sha256(champion),
        "asu_spec_hash": FROZEN_SPEC_HASH,
        "config": config.as_dict(),
        "config_sha256": hashlib.sha256(
            json.dumps(config.as_dict(), separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "engine_hashes": engine_hashes(),
        "source_hashes": source_hashes(),
        "available_baselines": list(matchups),
        "screens": screens,
        "screen_passed": screen_passed,
        "asu_rollout_screen": rollout_screen,
        "asu_rollout_screen_passed": rollout_screen_passed,
        "full": full,
        "passed": passed,
        "label": "champion" if passed else "candidate",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_release(
    candidate: str | Path,
    gate_report: dict,
    config: BenchmarkConfig,
    *,
    release_root: str | Path = BENCH_ROOT / "releases",
) -> Path:
    config.assert_frozen_v1()
    candidate = Path(candidate)
    expected_config_hash = hashlib.sha256(
        json.dumps(config.as_dict(), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    required = (
        gate_report.get("schema") == 1
        and gate_report.get("ruleset") == config.ruleset
        and gate_report.get("passed") is True
        and gate_report.get("label") == "champion"
        and gate_report.get("candidate_sha256") == _sha256(candidate)
        and gate_report.get("asu_spec_hash") == FROZEN_SPEC_HASH
        and gate_report.get("asu_rollout_screen_passed") is True
        and gate_report.get("config") == config.as_dict()
        and gate_report.get("config_sha256") == expected_config_hash
        and gate_report.get("engine_hashes") == engine_hashes()
        and gate_report.get("source_hashes") == source_hashes()
    )
    names = gate_report.get("available_baselines", [])
    full = gate_report.get("full", {})
    if not required or not names or set(full) != set(names):
        raise ValueError("Gate report is not bound to this candidate and frozen configuration")
    if not all(full_gate_passes(MatchSummary(**full[name]), config) for name in names):
        raise ValueError("Gate report contains a matchup that does not pass the full gate")
    release = Path(release_root) / RELEASE_TAG
    if release.exists():
        raise FileExistsError(f"Immutable release already exists: {release}")
    temporary = release.with_name(release.name + ".tmp")
    temporary.mkdir(parents=True)
    model_path = temporary / "monopolyzero-ppo-plus-v2-v1.pt"
    shutil.copy2(candidate, model_path)
    model_hash = _sha256(model_path)
    _atomic_json(temporary / "gate-report.json", gate_report)
    _atomic_json(
        temporary / "manifest.json",
        {
            "tag": RELEASE_TAG,
            "ruleset": config.ruleset,
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "players": config.players,
            "scope": gate_report["scope"],
            "model": model_path.name,
            "model_sha256": model_hash,
            "asu_spec_hash": FROZEN_SPEC_HASH,
            "engine_hashes": engine_hashes(),
            "source_hashes": source_hashes(),
            "config": config.as_dict(),
        },
    )
    (temporary / "SHA256SUMS").write_text(f"{model_hash}  {model_path.name}\n")
    release.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, release)
    return release


def gate_run_directory(run_dir: str | Path) -> tuple[dict, Path | None]:
    run_dir = Path(run_dir)
    config = BenchmarkConfig.load(run_dir / "config.json")
    promoted = sorted((run_dir / "snapshots").glob("promoted_*.pt"))
    if len(promoted) < 2:
        raise ValueError("No post-bootstrap promoted candidate is available for the king gate")
    candidate = promoted[-1]
    report = run_gate(candidate, config=config)
    _atomic_json(run_dir / "reports" / f"gate_{candidate.stem}.json", report)
    _atomic_json(run_dir / "status.json", {"status": report["label"], "candidate": str(candidate)})
    release = freeze_release(candidate, report, config) if report["passed"] else None
    return report, release
