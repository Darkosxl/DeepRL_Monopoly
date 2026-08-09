from __future__ import annotations

import json

import pytest

from monopoly_bench.adapters import SLMAdapter, available_baselines
from monopoly_bench.arena import balanced_single_seats, balanced_team_seats, wilson_lower
from monopoly_bench.config import BenchmarkConfig
from monopoly_bench.contracts import MatchSummary
from monopoly_bench.engine import SharedGame
from monopoly_bench.ladder import freeze_release, full_gate_passes, promotion_passes, run_gate
from monopoly_bench.model import MonopolyZeroNet


def _summary(games=100, wins=50, lower=0.41, latency=1.9) -> MatchSummary:
    return MatchSummary(games, games, wins, games - wins, wins / games, lower, 0, 0, 0, latency)


def test_every_repository_baseline_adapter_is_legal() -> None:
    game = SharedGame.new(3, max_rounds=1)
    actor = game.env.whose_turn()
    baselines = available_baselines()
    assert set(baselines) == {
        "TheHoarder", "TheDealMaker", "TheGambler", "TheBuilder",
        "TheBlocker", "TheRailBaron", "asu_value_v1", "ppo_v2", "cfr_v2",
    }
    legal = game.env.get_allowed_actions(actor)
    assert all(policy.choose_action(game, actor, 7).action in legal for policy in baselines.values())


def test_slm_callable_uses_canonical_contract_and_marks_fallbacks() -> None:
    game = SharedGame.new(3, max_rounds=1)
    actor = game.env.whose_turn()
    valid = SLMAdapter(lambda prompt: '{"action":"end_turn"}').choose_action(game, actor, 1)
    invalid = SLMAdapter(lambda prompt: "not json").choose_action(game, actor, 1)
    assert valid.action == 1 and not valid.fallback
    assert invalid.action in game.env.get_allowed_actions(actor) and invalid.fallback


def test_seat_balancing_and_wilson_boundaries() -> None:
    teams = balanced_team_seats(120)
    singles = balanced_single_seats(100)
    assert all(teams.count(team) == 20 for team in set(teams))
    assert all(singles.count(seat) == 25 for seat in set(singles))
    assert wilson_lower(50, 100) > 0.40


def test_promotion_and_freeze_predicates_fail_closed() -> None:
    config = BenchmarkConfig()
    assert promotion_passes(_summary(120, 66, 0.451), config)
    assert not promotion_passes(_summary(120, 66, 0.45), config)
    assert full_gate_passes(_summary(), config)
    assert not full_gate_passes(_summary(latency=2.01), config)
    unsafe = MatchSummary(100, 99, 100, -1, 1.0, 0.9, 0, 0, 0, 0.1)
    assert not full_gate_passes(unsafe, config)


def test_gate_runs_screen_then_full_with_gate_seeds(monkeypatch, bench_tmp) -> None:
    import monopoly_bench.ladder as ladder

    calls = []
    monkeypatch.setattr(ladder, "available_baselines", lambda *args, **kwargs: {"baseline": object()})

    def fake_evaluate(champion, baseline, *, games, config, seed_base):
        calls.append((games, seed_base))
        return _summary(games, games, 0.9, 0.1)

    monkeypatch.setattr(ladder, "evaluate_baseline", fake_evaluate)
    config = BenchmarkConfig()
    candidate = bench_tmp / "candidate.pt"
    MonopolyZeroNet().save_inference(candidate)
    report = run_gate(candidate, config=config)
    assert report["passed"] and report["label"] == "champion"
    assert calls == [(20, config.seeds.gate), (100, config.seeds.gate + 100_000)]


def test_gate_adds_mixed_asu_matchup_and_rollout_screen(monkeypatch, bench_tmp) -> None:
    import monopoly_bench.ladder as ladder

    baselines = {
        "asu_value_v1": object(),
        "TheDealMaker": object(),
        "TheGambler": object(),
    }
    monkeypatch.setattr(ladder, "available_baselines", lambda *args, **kwargs: baselines)
    calls = []

    def fake_evaluate(champion, baseline, *, games, config, seed_base):
        calls.append((games, seed_base))
        return _summary(games, games, 0.9, 0.1)

    monkeypatch.setattr(ladder, "evaluate_baseline", fake_evaluate)
    config = BenchmarkConfig()
    candidate = bench_tmp / "candidate.pt"
    MonopolyZeroNet().save_inference(candidate)
    report = run_gate(candidate, config=config)
    assert report["passed"] and report["asu_rollout_screen_passed"]
    assert "mixed_asu_dealmaker_gambler" in report["available_baselines"]
    assert (20, config.seeds.gate + 50_000) in calls
    assert len(calls) == 9


def test_only_a_model_bound_frozen_gate_can_make_an_immutable_release(monkeypatch, bench_tmp) -> None:
    import monopoly_bench.ladder as ladder

    model_path = bench_tmp / "candidate.pt"
    MonopolyZeroNet().save_inference(model_path)
    config = BenchmarkConfig()
    with pytest.raises(ValueError):
        freeze_release(model_path, {"passed": False}, config, release_root=bench_tmp / "releases")
    monkeypatch.setattr(ladder, "available_baselines", lambda *args, **kwargs: {"baseline": object()})
    monkeypatch.setattr(
        ladder,
        "evaluate_baseline",
        lambda champion, baseline, *, games, config, seed_base: _summary(games, games, 0.9, 0.1),
    )
    report = run_gate(model_path, config=config)
    release = freeze_release(model_path, report, config, release_root=bench_tmp / "releases")
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["ruleset"] == "ppo-plus-v2"
    assert (release / "SHA256SUMS").read_text().split()[0] == manifest["model_sha256"]
    with pytest.raises(FileExistsError):
        freeze_release(model_path, report, config, release_root=bench_tmp / "releases")


def test_weakened_config_cannot_enter_the_king_gate(bench_tmp) -> None:
    candidate = bench_tmp / "candidate.pt"
    MonopolyZeroNet().save_inference(candidate)
    weakened = BenchmarkConfig(max_rounds=1)
    with pytest.raises(ValueError, match="exact frozen"):
        run_gate(candidate, config=weakened)
