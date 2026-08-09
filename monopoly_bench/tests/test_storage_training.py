from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from monopoly_bench.config import (
    BenchmarkConfig,
    ResourceConfig,
    SearchConfig,
    TrainingConfig,
)
from monopoly_bench.contracts import ReplayPosition
from monopoly_bench.model import MonopolyZeroNet
from monopoly_bench.storage import (
    CheckpointManager,
    ReplayBuffer,
    pending_replay_descriptor,
)
from monopoly_bench.training import (
    Trainer,
    _seed_all,
    bootstrap_asu_expert,
    bootstrap_value_head,
    generate_population_games,
    load_asu_examples,
    population_jobs,
    save_asu_examples,
)


PPO = "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"


def _position() -> ReplayPosition:
    mask = np.zeros(2_958, dtype=np.bool_)
    mask[[1, 3]] = True
    return ReplayPosition(
        state=np.linspace(0, 1, 300, dtype=np.float32),
        legal_mask=mask,
        visits={1: 5, 3: 2},
        q_values={1: (1.0, 0.0, 0.0, 0.0), 3: (0.4, 0.3, 0.2, 0.1)},
        selected_action=1,
        value=(0.4, 0.3, 0.2, 0.1),
        outcome=(1.0, 0.0, 0.0, 0.0),
        actor_id=0,
        game_id=7,
    )


def _expert_examples(count: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    masks = np.zeros((count, 2_958), dtype=np.bool_)
    masks[:, [1, 2]] = True
    return {
        "states": rng.normal(size=(count, 300)).astype(np.float32),
        "legal_masks": masks,
        "selected_actions": np.where(np.arange(count) % 2, 1, 2).astype(np.int64),
        "actors": np.arange(count, dtype=np.int64) % 4,
        "outcomes": np.eye(4, dtype=np.float32)[np.arange(count) % 4],
        "teachers": (np.arange(count) == 0).astype(np.uint8),
    }


def test_replay_round_trip(bench_tmp) -> None:
    replay = ReplayBuffer(bench_tmp / "replay", capacity=3, create=True)
    replay.append(_position())
    restored = ReplayBuffer(bench_tmp / "replay")
    record = restored.records(game_id=7)[0]
    assert record.visits == {1: 5, 3: 2}
    assert record.q_values[3] == pytest.approx(_position().q_values[3])
    assert np.array_equal(record.legal_mask, _position().legal_mask)
    batch = restored.sample(1, np.random.default_rng(1))
    assert batch["states"].shape == (1, 300)
    assert batch["legal_masks"].shape == (1, 2_958)


def test_atomic_checkpoint_restores_every_rng(bench_tmp) -> None:
    replay = ReplayBuffer(bench_tmp / "replay", capacity=3, create=True)
    replay.append(_position())
    model = MonopolyZeroNet()
    optimizer = torch.optim.AdamW(model.parameters())
    config = BenchmarkConfig()
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    path = bench_tmp / "checkpoint.pt"
    CheckpointManager.save(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        replay=replay,
        generation=3,
        league={"incumbent": "x"},
        promotion_history=[{"passed": False}],
    )
    expected = random.random(), np.random.rand(), torch.rand(3)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    resumed = CheckpointManager.load(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        replay=replay,
    )
    actual = random.random(), np.random.rand(), torch.rand(3)
    assert resumed.generation == 3
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_replay_transaction_recovers_both_sides_of_a_crash(bench_tmp) -> None:
    run_dir = bench_tmp / "run"
    replay = ReplayBuffer(run_dir / "replay", capacity=3, create=True)
    staging = ReplayBuffer(run_dir / "pending/generation_0001", capacity=1, create=True)
    staging.append(_position())
    model = MonopolyZeroNet()
    optimizer = torch.optim.AdamW(model.parameters())
    config = BenchmarkConfig()
    descriptor = pending_replay_descriptor(staging, replay, run_dir)
    checkpoint = run_dir / "checkpoints/generation_0001.pt"
    CheckpointManager.save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        replay=replay,
        generation=1,
        league={},
        promotion_history=[],
        pending_replay=descriptor,
    )

    # Crash before memmap commit: recovery rolls the journal forward once.
    CheckpointManager.reconcile_replay(checkpoint, replay, run_dir)
    assert len(replay) == 1 and replay.total == 1
    CheckpointManager.reconcile_replay(checkpoint, replay, run_dir)
    assert len(replay) == 1 and replay.total == 1
    resumed = CheckpointManager.load(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        replay=replay,
    )
    assert resumed.generation == 1 and resumed.pending_replay == descriptor


def test_value_bootstrap_never_changes_ppo_actor() -> None:
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith("value_head")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    examples = {
        "states": np.random.default_rng(1).normal(size=(8, 300)).astype(np.float32),
        "actors": np.arange(8, dtype=np.int64) % 4,
        "outcomes": np.eye(4, dtype=np.float32)[np.arange(8) % 4],
    }
    bootstrap_value_head(model, optimizer, scaler, examples, updates=2, batch_size=4, gradient_clip=1, seed=4)
    after = model.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_asu_expert_round_trip_and_bootstrap_updates_policy(tmp_path) -> None:
    source = tmp_path / "asu.npz"
    save_asu_examples(source, _expert_examples())
    examples = load_asu_examples((source,))
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    before = model.policy_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    stats = bootstrap_asu_expert(
        model,
        optimizer,
        scaler,
        examples,
        updates=2,
        batch_size=4,
        gradient_clip=1,
        seed=4,
    )
    assert not torch.equal(before, model.policy_head.weight)
    assert stats["head_only_updates"] == 1
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_fresh_value_head_initialization_uses_the_run_seed() -> None:
    _seed_all(BenchmarkConfig().seeds.bootstrap)
    first = MonopolyZeroNet().value_head.state_dict()
    _seed_all(BenchmarkConfig().seeds.bootstrap)
    second = MonopolyZeroNet().value_head.state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_tiny_generation_and_resume(bench_tmp) -> None:
    config = BenchmarkConfig(
        max_rounds=1,
        search=SearchConfig(simulations=1, max_depth=8),
        training=TrainingConfig(
            bootstrap_games=4,
            value_updates=1,
            games_per_generation=4,
            workers=1,
            replay_capacity=5_000,
            updates_per_generation=1,
            batch_size=2,
            promoted_snapshot_limit=2,
            promotion_games=6,
            promotion_win_rate=1.0,
            promotion_wilson_lower=0.99,
        ),
        resources=ResourceConfig(
            soft_rss_gib=100,
            hard_rss_gib=101,
            min_available_ram_gib=0,
            reserved_vram_gib=0,
            min_free_disk_gib=0,
        ),
    )
    expert = bench_tmp / "expert.npz"
    save_asu_examples(expert, _expert_examples())
    trainer = Trainer(
        bench_tmp / "run",
        PPO,
        config=config,
        device="cpu",
        asu_expert_data=(expert,),
    )
    report = trainer.run(1)[0]
    assert report["games"] == 4 and report["positions"] > 0
    resumed = Trainer(bench_tmp / "run", PPO, config=config, device="cpu")
    assert resumed.generation == 1
    assert len(resumed.replay) == report["replay_size"]


def test_population_uses_four_spawned_cpu_workers() -> None:
    config = BenchmarkConfig(
        max_rounds=1,
        search=SearchConfig(simulations=1, max_depth=8),
        training=TrainingConfig(
            bootstrap_games=1,
            value_updates=1,
            games_per_generation=4,
            workers=4,
            replay_capacity=100,
            updates_per_generation=1,
            batch_size=1,
            promotion_games=6,
        ),
    )
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    games = generate_population_games(
        model,
        config=config,
        generation=1,
        snapshots=[],
        ppo_checkpoint=PPO,
        cfr_checkpoint="artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz",
    )
    assert len(games) == 4
    assert all(game.completed and not game.crashes for game in games)


def test_population_reserves_half_of_baseline_games_for_asu() -> None:
    config = BenchmarkConfig()
    jobs = population_jobs(
        generation=1,
        config=config,
        snapshots=[],
        baseline_names=["asu_value_v1", "TheDealMaker"],
    )
    baseline_jobs = [job for job in jobs if job["category"] == "baseline"]
    assert len(baseline_jobs) == 8
    assert sum(job["baseline"] == "asu_value_v1" for job in baseline_jobs) == 4
