"""PPO bootstrap, population self-play, replay learning, and resumable runs."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ASU_FROZEN_TEACHER import FROZEN_SPEC_HASH

from .adapters import (
    ASUAdapter,
    FixedAdapter,
    PPOAdapter,
    SearchAdapter,
    available_baselines,
    fixed_baselines,
)
from .arena import balanced_single_seats, balanced_team_seats, play_game
from .colab import ColabLauncher, build_resume_bundle
from .config import BenchmarkConfig, SearchConfig
from .contracts import GameResult
from .engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, RULESET_VERSION, SharedGame, legal_mask
from .model import MonopolyZeroNet
from .storage import (
    CheckpointManager,
    ReplayBuffer,
    _atomic_json,
    pending_replay_descriptor,
)
from .watchdog import ResourceLimitReached, ResourceWatchdog
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def _relative_outcomes(outcomes: torch.Tensor, actors: torch.Tensor) -> torch.Tensor:
    orders = torch.stack(
        [
            torch.tensor(
                [int(actor), *(pid for pid in range(NUM_PLAYERS) if pid != int(actor))],
                device=outcomes.device,
            )
            for actor in actors
        ]
    )
    return outcomes.gather(1, orders)


def train_step(
    model: MonopolyZeroNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, np.ndarray],
    gradient_clip: float,
    *,
    value_only: bool = False,
) -> dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    outcomes = torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device)
    actors = torch.as_tensor(batch["actors"], dtype=torch.long, device=device)
    targets = _relative_outcomes(outcomes, actors)
    masks = None
    if not value_only:
        masks = torch.as_tensor(batch["legal_masks"], dtype=torch.bool, device=device)

    optimizer.zero_grad(set_to_none=True)
    amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits, values = model(states, masks)
        value_loss = -(targets * values.clamp_min(1e-8).log()).sum(dim=1).mean()
        if value_only:
            policy_loss = torch.zeros((), device=device)
        else:
            actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
            counts = torch.as_tensor(batch["visit_counts"], dtype=torch.float32, device=device)
            lengths = torch.as_tensor(batch["lengths"], dtype=torch.long, device=device)
            slots = torch.arange(actions.shape[1], device=device).unsqueeze(0)
            weights = counts * (slots < lengths.unsqueeze(1))
            totals = weights.sum(dim=1, keepdim=True)
            if (totals <= 0).any():
                raise ValueError("A replay policy target has no MCTS visits")
            weights = weights / totals
            gathered = torch.log_softmax(logits, dim=1).gather(1, actions.clamp_min(0))
            gathered = torch.where(weights > 0, gathered, torch.zeros_like(gathered))
            policy_loss = -(weights * gathered).sum(dim=1).mean()
        loss = policy_loss + value_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "gradient_norm": float(gradient_norm),
    }


def expert_train_step(
    model: MonopolyZeroNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, np.ndarray],
    gradient_clip: float,
) -> dict[str, float]:
    """Imitate a frozen ASU action while learning value from the real winner."""
    model.train()
    device = next(model.parameters()).device
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["legal_masks"], dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["selected_actions"], dtype=torch.long, device=device)
    outcomes = torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device)
    actors = torch.as_tensor(batch["actors"], dtype=torch.long, device=device)
    targets = _relative_outcomes(outcomes, actors)

    optimizer.zero_grad(set_to_none=True)
    amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits, values = model(states, masks)
        policy_loss = F.cross_entropy(logits, actions)
        value_loss = -(targets * values.clamp_min(1e-8).log()).sum(dim=1).mean()
        loss = policy_loss + value_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "gradient_norm": float(gradient_norm),
    }


_EXPERT_FIELDS = (
    "states",
    "legal_masks",
    "selected_actions",
    "actors",
    "outcomes",
    "teachers",
)


def _validate_asu_examples(examples: dict[str, np.ndarray]) -> None:
    missing = set(_EXPERT_FIELDS) - set(examples)
    if missing:
        raise ValueError(f"ASU expert data is missing fields: {sorted(missing)}")
    count = len(examples["states"])
    expected = {
        "states": (count, 300),
        "legal_masks": (count, 2_958),
        "selected_actions": (count,),
        "actors": (count,),
        "outcomes": (count, NUM_PLAYERS),
        "teachers": (count,),
    }
    actual = {name: tuple(examples[name].shape) for name in _EXPERT_FIELDS}
    if count < 1 or actual != expected:
        raise ValueError(f"Bad ASU expert shapes: {actual}; expected {expected}")
    if not np.isfinite(examples["states"]).all() or not np.isfinite(examples["outcomes"]).all():
        raise ValueError("ASU expert data contains non-finite values")
    actions = examples["selected_actions"].astype(np.int64, copy=False)
    if (actions < 0).any() or (actions >= 2_958).any():
        raise ValueError("ASU expert data contains an out-of-range action")
    if not examples["legal_masks"][np.arange(count), actions].all():
        raise ValueError("ASU expert data contains an illegal selected action")


def save_asu_examples(path: str | Path, examples: dict[str, np.ndarray]) -> Path:
    _validate_asu_examples(examples)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{name: examples[name] for name in _EXPERT_FIELDS},
            ruleset=np.asarray(RULESET_VERSION),
            asu_spec_hash=np.asarray(FROZEN_SPEC_HASH),
        )
    os.replace(temporary, destination)
    return destination


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asu_examples(paths: Iterable[str | Path]) -> dict[str, np.ndarray]:
    groups = []
    for source in paths:
        with np.load(Path(source), allow_pickle=False) as payload:
            ruleset = str(payload["ruleset"].item())
            spec_hash = str(payload["asu_spec_hash"].item())
            if (ruleset, spec_hash) != (RULESET_VERSION, FROZEN_SPEC_HASH):
                raise ValueError(
                    f"Incompatible ASU expert data: {(ruleset, spec_hash)}"
                )
            group = {name: np.asarray(payload[name]).copy() for name in _EXPERT_FIELDS}
        _validate_asu_examples(group)
        groups.append(group)
    if not groups:
        raise ValueError("At least one ASU expert shard is required")
    merged = {
        name: np.concatenate([group[name] for group in groups], axis=0)
        for name in _EXPERT_FIELDS
    }
    _validate_asu_examples(merged)
    return merged


def collect_asu_examples(
    *,
    games: int,
    seed_base: int,
    max_rounds: int,
    rollout_positions: int = 0,
) -> dict[str, np.ndarray]:
    """Collect seat-balanced ASU actions with physical final-winner labels."""
    if games < 1 or rollout_positions < 0:
        raise ValueError("ASU games must be positive and rollout positions nonnegative")
    value = ASUAdapter()
    rollout = ASUAdapter(rollout=True)
    fixed = [FixedAdapter(agent_class) for agent_class in FP_AGENT_CLASSES[:3]]
    rollout_games = {
        min(games - 1, index * games // rollout_positions)
        for index in range(rollout_positions)
    } if rollout_positions else set()
    collected: dict[str, list] = {name: [] for name in _EXPERT_FIELDS}

    for game_index in range(games):
        seed = seed_base + game_index
        game = SharedGame.new(seed, max_rounds)
        teacher_seat = game_index % NUM_PLAYERS
        opponent_seats = [seat for seat in range(NUM_PLAYERS) if seat != teacher_seat]
        opponents = dict(zip(opponent_seats, fixed))
        pending = []
        used_rollout = False
        for step in range(max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN):
            if game.env.done:
                break
            actor = game.env.whose_turn()
            legal = tuple(game.env.get_allowed_actions(actor))
            if len(legal) == 1:
                action = legal[0]
            elif actor == teacher_seat:
                use_rollout = game_index in rollout_games and not used_rollout
                policy = rollout if use_rollout else value
                decision = policy.choose_action(game, actor, seed * 1_000_003 + step)
                action = decision.action
                used_rollout |= use_rollout
                pending.append(
                    (
                        game.env._get_state(actor),
                        legal_mask(legal),
                        action,
                        actor,
                        int(use_rollout),
                    )
                )
            else:
                action = opponents[actor].choose_action(
                    game, actor, seed * 1_000_003 + step
                ).action
            if action not in legal:
                raise RuntimeError(f"ASU bootstrap produced illegal action {action}")
            game.step(action)
        if not game.env.done:
            raise RuntimeError(f"ASU bootstrap game {game_index} did not complete")
        outcome = np.zeros(NUM_PLAYERS, dtype=np.float32)
        outcome[game.env.winner()] = 1.0
        for state, mask, action, actor, teacher in pending:
            collected["states"].append(state)
            collected["legal_masks"].append(mask)
            collected["selected_actions"].append(action)
            collected["actors"].append(actor)
            collected["outcomes"].append(outcome)
            collected["teachers"].append(teacher)

    examples = {
        "states": np.asarray(collected["states"], dtype=np.float32),
        "legal_masks": np.asarray(collected["legal_masks"], dtype=np.bool_),
        "selected_actions": np.asarray(collected["selected_actions"], dtype=np.int64),
        "actors": np.asarray(collected["actors"], dtype=np.int64),
        "outcomes": np.asarray(collected["outcomes"], dtype=np.float32),
        "teachers": np.asarray(collected["teachers"], dtype=np.uint8),
    }
    _validate_asu_examples(examples)
    return examples


def bootstrap_asu_expert(
    model: MonopolyZeroNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    examples: dict[str, np.ndarray],
    *,
    updates: int,
    batch_size: int,
    gradient_clip: float,
    seed: int,
) -> dict[str, float]:
    """Train both heads first, then the full PPO-initialized network."""
    _validate_asu_examples(examples)
    model.freeze_trunk()
    rng = np.random.default_rng(seed)
    head_updates = max(1, updates // 4)
    last = {}
    for update in range(updates):
        if update == head_updates:
            model.unfreeze_all()
        indices = rng.choice(
            len(examples["states"]),
            size=min(batch_size, len(examples["states"])),
            replace=False,
        )
        last = expert_train_step(
            model,
            optimizer,
            scaler,
            {name: values[indices] for name, values in examples.items()},
            gradient_clip,
        )
    model.unfreeze_all()
    return {**last, "head_only_updates": float(min(head_updates, updates))}


def collect_bootstrap_examples(
    checkpoint: str | Path,
    *,
    games: int,
    seed_base: int,
    max_rounds: int,
) -> dict[str, np.ndarray]:
    """Collect actor-perspective PPO positions with physical winner labels."""
    ppo = PPOAdapter(checkpoint, stochastic=True)
    fixed = list(fixed_baselines().values())
    states: list[np.ndarray] = []
    actors: list[int] = []
    outcomes: list[tuple[float, ...]] = []
    for game_index in range(games):
        seed = seed_base + game_index
        game = SharedGame.new(seed, max_rounds)
        ppo_seat = game_index % NUM_PLAYERS
        pending: list[tuple[np.ndarray, int]] = []
        for step in range(max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN):
            if game.env.done:
                break
            player_id = game.env.whose_turn()
            legal = tuple(game.env.get_allowed_actions(player_id))
            if len(legal) == 1:
                action = legal[0]
            else:
                policy = ppo if player_id == ppo_seat else fixed[(game_index + player_id) % len(fixed)]
                decision = policy.choose_action(game, player_id, seed * 1_000_003 + step)
                action = decision.action
                if player_id == ppo_seat:
                    pending.append((game.env._get_state(player_id), player_id))
            if action not in legal:
                raise RuntimeError("Bootstrap policy produced an illegal action")
            game.step(action)
        if not game.env.done:
            raise RuntimeError(f"Bootstrap game {game_index} did not complete")
        outcome = [0.0] * NUM_PLAYERS
        outcome[game.env.winner()] = 1.0
        for state, actor in pending:
            states.append(state)
            actors.append(actor)
            outcomes.append(tuple(outcome))
    if not states:
        raise RuntimeError("Bootstrap games yielded no non-forced PPO positions")
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actors": np.asarray(actors, dtype=np.int64),
        "outcomes": np.asarray(outcomes, dtype=np.float32),
    }


def bootstrap_value_head(
    model: MonopolyZeroNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    examples: dict[str, np.ndarray],
    *,
    updates: int,
    batch_size: int,
    gradient_clip: float,
    seed: int,
) -> dict[str, float]:
    model.freeze_policy()
    rng = np.random.default_rng(seed)
    last = {}
    for _ in range(updates):
        indices = rng.choice(len(examples["states"]), size=min(batch_size, len(examples["states"])), replace=False)
        last = train_step(
            model,
            optimizer,
            scaler,
            {name: values[indices] for name, values in examples.items()},
            gradient_clip,
            value_only=True,
        )
    model.unfreeze_all()
    return last


def _load_model(path: str | Path | None, state: dict[str, torch.Tensor] | None = None) -> MonopolyZeroNet:
    if path is not None:
        return MonopolyZeroNet.load_inference(path)
    model = MonopolyZeroNet()
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _population_worker(payload: dict) -> list[GameResult]:
    torch.set_num_threads(1)
    current = _load_model(None, payload["model_state"])
    search_config = SearchConfig(**payload["search_config"])
    baselines = available_baselines(payload["ppo_checkpoint"], payload["cfr_checkpoint"])
    snapshot_cache: dict[str, MonopolyZeroNet] = {}
    results = []
    for job in payload["jobs"]:
        category = job["category"]
        target = set(job["current_seats"])
        policies: dict[int, object] = {}
        for seat in target:
            policies[seat] = SearchAdapter(current, search_config, self_play=True)
        if category == "self":
            for seat in range(NUM_PLAYERS):
                policies.setdefault(seat, SearchAdapter(current, search_config, self_play=True))
        elif category == "snapshot":
            snapshot_path = job.get("snapshot")
            if snapshot_path:
                snapshot_cache.setdefault(snapshot_path, _load_model(snapshot_path))
                opponent_model = snapshot_cache[snapshot_path]
            else:
                opponent_model = current
            for seat in set(range(NUM_PLAYERS)) - target:
                policies[seat] = SearchAdapter(opponent_model, search_config, self_play=False)
        else:
            baseline = baselines[job["baseline"]]
            for seat in set(range(NUM_PLAYERS)) - target:
                policies[seat] = baseline
        results.append(
            play_game(
                game_id=job["game_id"],
                seed=job["seed"],
                policies=policies,
                max_rounds=payload["max_rounds"],
                record_seats=target,
            )
        )
    return results


def population_jobs(
    *,
    generation: int,
    config: BenchmarkConfig,
    snapshots: list[str],
    baseline_names: list[str],
) -> list[dict]:
    count = config.training.games_per_generation
    if count % 4:
        raise ValueError("Population fractions require a multiple of four games")
    jobs = []
    baseline_count = count // 4
    asu_count = max(1, baseline_count // 2)
    team_seats = balanced_team_seats(baseline_count)
    single_seats = balanced_single_seats(baseline_count)
    other_baselines = [name for name in baseline_names if name != "asu_value_v1"]
    for index in range(count):
        if index < count // 2:
            category, seats = "self", set(range(NUM_PLAYERS))
        elif index < 3 * count // 4:
            local = index - count // 2
            category, seats = "snapshot", set(team_seats[local])
        else:
            local = index - 3 * count // 4
            category, seats = "baseline", set(single_seats[local])
        job = {
            "category": category,
            "current_seats": sorted(seats),
            "game_id": generation * 100_000 + index,
            "seed": config.seeds.self_play + generation * count + index,
        }
        if category == "snapshot":
            job["snapshot"] = snapshots[local % len(snapshots)] if snapshots else None
        if category == "baseline":
            job["baseline"] = (
                "asu_value_v1"
                if local < asu_count or not other_baselines
                else other_baselines[(local - asu_count) % len(other_baselines)]
            )
        jobs.append(job)
    return jobs


def generate_population_games(
    model: MonopolyZeroNet,
    *,
    config: BenchmarkConfig,
    generation: int,
    snapshots: list[str],
    ppo_checkpoint: str | Path,
    cfr_checkpoint: str | Path,
) -> list[GameResult]:
    baseline_names = list(available_baselines(ppo_checkpoint, cfr_checkpoint))
    jobs = population_jobs(
        generation=generation,
        config=config,
        snapshots=snapshots[-config.training.promoted_snapshot_limit :],
        baseline_names=baseline_names,
    )
    workers = config.training.workers
    chunks = [jobs[index::workers] for index in range(workers)]
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payloads = [
        {
            "model_state": state,
            "search_config": asdict(config.search),
            "max_rounds": config.max_rounds,
            "ppo_checkpoint": str(ppo_checkpoint),
            "cfr_checkpoint": str(cfr_checkpoint),
            "jobs": chunk,
        }
        for chunk in chunks
    ]
    if workers == 1:
        nested = [_population_worker(payloads[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            nested = pool.map(_population_worker, payloads)
    return sorted((result for group in nested for result in group), key=lambda result: result.game_id)


class Trainer:
    def __init__(
        self,
        run_dir: str | Path,
        bootstrap_ppo: str | Path,
        *,
        config: BenchmarkConfig | None = None,
        device: str = "auto",
        fallback_colab: bool = False,
        asu_expert_data: Iterable[str | Path] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.config = config or BenchmarkConfig()
        self.bootstrap_ppo = Path(bootstrap_ppo)
        self.cfr_checkpoint = self.bootstrap_ppo.parents[1] / "cfr_ppo_plus/cfr_full_game_v2.pkl.gz"
        if not self.cfr_checkpoint.exists():
            self.cfr_checkpoint = self.run_dir.parents[1] / "artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz"
        self.fallback_colab = fallback_colab
        self.device = _device(device)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "candidates", "snapshots", "reports", "colab"):
            (self.run_dir / name).mkdir(exist_ok=True)
        self.expert_path = self.run_dir / "asu_expert.npz"
        expert_sources = tuple(asu_expert_data or ())

        config_path = self.run_dir / "config.json"
        if config_path.exists() and BenchmarkConfig.load(config_path) != self.config:
            raise ValueError("Run directory contains a different configuration")
        self.config.save(config_path)
        replay_path = self.run_dir / "replay"
        self.replay = ReplayBuffer(
            replay_path,
            self.config.training.replay_capacity,
            create=not (replay_path / "metadata.json").exists(),
        )
        _seed_all(self.config.seeds.bootstrap)
        self.model = MonopolyZeroNet().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.generation = 0
        self.league: dict = {"incumbent": None, "snapshots": [], "status": "candidate"}
        self.promotion_history: list[dict] = []
        self.watchdog = ResourceWatchdog(self.run_dir, self.config.resources)

        checkpoints = sorted((self.run_dir / "checkpoints").glob("generation_*.pt"))
        if checkpoints:
            CheckpointManager.reconcile_replay(checkpoints[-1], self.replay, self.run_dir)
            resumed = CheckpointManager.load(
                checkpoints[-1],
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                config=self.config,
                replay=self.replay,
            )
            self.generation = resumed.generation
            self.league = resumed.league
            self.promotion_history = resumed.promotion_history
            self.expert = self._prepare_expert(expert_sources)
            if self.league.get("asu_expert_sha256") != _file_sha256(self.expert_path):
                raise ValueError("ASU expert data differs from the checkpointed run")
            self._recover_release()
        else:
            self.expert = self._prepare_expert(expert_sources)
            self._bootstrap()

    def _prepare_expert(
        self,
        sources: tuple[str | Path, ...],
    ) -> dict[str, np.ndarray]:
        if self.expert_path.exists():
            return load_asu_examples((self.expert_path,))
        if sources:
            examples = load_asu_examples(sources)
        else:
            rollout_positions = (
                min(
                    self.config.training.asu_rollout_bootstrap_positions,
                    self.config.training.bootstrap_games,
                )
                if self.config.max_rounds == 200
                else 0
            )
            examples = collect_asu_examples(
                games=self.config.training.bootstrap_games,
                seed_base=self.config.seeds.bootstrap,
                max_rounds=self.config.max_rounds,
                rollout_positions=rollout_positions,
            )
        save_asu_examples(self.expert_path, examples)
        return examples

    def _checkpoint(self, pending_replay: dict | None = None) -> Path:
        path = self.run_dir / "checkpoints" / f"generation_{self.generation:04d}.pt"
        CheckpointManager.save(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            config=self.config,
            replay=self.replay,
            generation=self.generation,
            league=self.league,
            promotion_history=self.promotion_history,
            pending_replay=pending_replay,
        )
        return path

    def _restore_latest(self) -> None:
        checkpoint = sorted((self.run_dir / "checkpoints").glob("generation_*.pt"))[-1]
        CheckpointManager.reconcile_replay(checkpoint, self.replay, self.run_dir)
        resumed = CheckpointManager.load(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            config=self.config,
            replay=self.replay,
        )
        self.generation = resumed.generation
        self.league = resumed.league
        self.promotion_history = resumed.promotion_history

    def _recover_release(self) -> None:
        if self.league.get("status") != "champion":
            return
        release = Path(self.league["release"])
        if release.exists():
            return
        from .ladder import freeze_release

        report = json.loads((self.run_dir / self.league["gate_report"]).read_text())
        freeze_release(self.run_dir / self.league["incumbent"], report, self.config)

    def _bootstrap(self) -> None:
        _seed_all(self.config.seeds.bootstrap)
        self.model.load_ppo_actor(self.bootstrap_ppo)
        stats = bootstrap_asu_expert(
            self.model,
            self.optimizer,
            self.scaler,
            self.expert,
            updates=self.config.training.value_updates,
            batch_size=self.config.training.batch_size,
            gradient_clip=self.config.training.gradient_clip,
            seed=self.config.seeds.bootstrap + self.config.training.bootstrap_games,
        )
        incumbent = self.run_dir / "snapshots" / "promoted_0000.pt"
        expert_hash = _file_sha256(self.expert_path)
        self.model.save_inference(
            incumbent,
            {
                "generation": 0,
                "bootstrap": "asu_value_v1",
                "asu_spec_hash": FROZEN_SPEC_HASH,
                "asu_expert_sha256": expert_hash,
            },
        )
        self.league = {
            "incumbent": str(incumbent.relative_to(self.run_dir)),
            "snapshots": [str(incumbent.relative_to(self.run_dir))],
            "status": "candidate",
            "asu_spec_hash": FROZEN_SPEC_HASH,
            "asu_expert_sha256": expert_hash,
        }
        _atomic_json(
            self.run_dir / "reports" / "bootstrap.json",
            {
                "teacher": "asu_value_v1",
                "asu_spec_hash": FROZEN_SPEC_HASH,
                "asu_expert_sha256": expert_hash,
                "games": self.config.training.bootstrap_games,
                "positions": len(self.expert["states"]),
                "rollout_positions": int(self.expert["teachers"].sum()),
                "updates": self.config.training.value_updates,
                **stats,
            },
        )
        self._checkpoint()

    def _sample_training_batch(
        self,
        staging: ReplayBuffer,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if not len(self.replay):
            return staging.sample(batch_size, rng)
        desired_new = int(rng.integers(2)) if batch_size == 1 else batch_size // 2
        new_count = min(len(staging), desired_new)
        old_count = min(len(self.replay), batch_size - new_count)
        new_count = min(len(staging), batch_size - old_count)
        parts = []
        if old_count:
            parts.append(self.replay.sample(old_count, rng))
        if new_count:
            parts.append(staging.sample(new_count, rng))
        return {
            name: np.concatenate([part[name] for part in parts], axis=0)
            for name in parts[0]
            if name != "indices"
        }

    def _sample_expert_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        count = len(self.expert["states"])
        indices = rng.choice(count, size=min(batch_size, count), replace=False)
        return {name: values[indices] for name, values in self.expert.items()}

    def _train_updates(
        self,
        batch_size: int,
        staging: ReplayBuffer,
        generation: int,
    ) -> dict[str, float]:
        rng = np.random.default_rng(self.config.seeds.self_play + generation)
        total_updates = self.config.training.updates_per_generation
        decay = self.config.training.expert_decay_generations
        expert_fraction = self.config.training.expert_fraction * max(
            0.0,
            1.0 - (generation - 1) / decay,
        )
        expert_updates = round(total_updates * expert_fraction)
        last = {}
        for update in range(total_updates):
            if update % 10 == 0:
                self.watchdog.check()
            use_expert = (
                (update + 1) * expert_updates // total_updates
                > update * expert_updates // total_updates
            )
            step = expert_train_step if use_expert else train_step
            batch = (
                self._sample_expert_batch(batch_size, rng)
                if use_expert
                else self._sample_training_batch(staging, batch_size, rng)
            )
            last = step(
                self.model,
                self.optimizer,
                self.scaler,
                batch,
                self.config.training.gradient_clip,
            )
        return {**last, "expert_updates": float(expert_updates)}

    def _handoff(self, reason: str) -> None:
        checkpoint = self._checkpoint()
        bundle = self.run_dir / "colab" / f"resume_generation_{self.generation:04d}.bundle"
        build_resume_bundle(self.run_dir, bundle)
        _atomic_json(
            self.run_dir / "reports" / "handoff.json",
            {"reason": reason, "checkpoint": str(checkpoint), "bundle": str(bundle)},
        )
        if self.fallback_colab:
            ColabLauncher().launch(
                self.run_dir,
                bootstrap_ppo=self.bootstrap_ppo,
                cfr_checkpoint=self.cfr_checkpoint,
            )

    def run_generation(self) -> dict:
        self.watchdog.check()
        next_generation = self.generation + 1
        snapshots = [str(self.run_dir / path) for path in self.league.get("snapshots", [])][-8:]
        games = generate_population_games(
            self.model,
            config=self.config,
            generation=next_generation,
            snapshots=snapshots,
            ppo_checkpoint=self.bootstrap_ppo,
            cfr_checkpoint=self.cfr_checkpoint,
        )
        failures = [
            {"game_id": game.game_id, "error": game.error}
            for game in games
            if not game.completed or game.crashes or game.illegal_actions
        ]
        if failures:
            raise RuntimeError(f"Population games failed closed: {failures[:5]}")
        positions = [position for game in games for position in game.positions]
        if not positions:
            raise RuntimeError("Population generation yielded no searchable positions")
        staging_path = self.run_dir / "pending" / f"generation_{next_generation:04d}"
        staging = ReplayBuffer(staging_path, capacity=len(positions), create=True)
        staging.append_many(positions)
        batch_size = self.config.training.batch_size
        oom_retried = False
        try:
            while True:
                try:
                    train_stats = self._train_updates(batch_size, staging, next_generation)
                    break
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    self._restore_latest()
                    if not oom_retried:
                        oom_retried = True
                        batch_size = max(1, batch_size // 2)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise ResourceLimitReached("cuda_oom", "Repeated CUDA OOM") from exc
        except ResourceLimitReached:
            self._restore_latest()
            raise

        self.generation = next_generation
        candidate = self.run_dir / "candidates" / f"generation_{self.generation:04d}.pt"
        model_metadata = {
            "generation": self.generation,
            "status": "candidate",
            "asu_spec_hash": FROZEN_SPEC_HASH,
            "asu_expert_sha256": self.league["asu_expert_sha256"],
        }
        self.model.save_inference(candidate, model_metadata)
        from .ladder import evaluate_promotion

        incumbent = self.run_dir / self.league["incumbent"]
        promotion, promoted = evaluate_promotion(
            candidate,
            incumbent,
            config=self.config,
            seed_base=(
                self.config.seeds.promotion
                + self.generation * self.config.training.promotion_games
            ),
        )
        promotion_entry = {
            "generation": self.generation,
            "candidate": str(candidate.relative_to(self.run_dir)),
            "incumbent": self.league["incumbent"],
            "passed": promoted,
            **promotion.as_dict(),
        }
        self.promotion_history.append(promotion_entry)
        release_request = None
        if promoted:
            snapshot = self.run_dir / "snapshots" / f"promoted_{self.generation:04d}.pt"
            self.model.save_inference(
                snapshot,
                {**model_metadata, "status": "promoted", "promoted": True},
            )
            relative = str(snapshot.relative_to(self.run_dir))
            self.league["incumbent"] = relative
            self.league["snapshots"] = (
                self.league.get("snapshots", []) + [relative]
            )[-self.config.training.promoted_snapshot_limit :]
            from .ladder import BENCH_ROOT, RELEASE_TAG, freeze_release, run_gate

            gate_report = run_gate(
                snapshot,
                config=self.config,
                ppo_checkpoint=self.bootstrap_ppo,
                cfr_checkpoint=self.cfr_checkpoint,
            )
            promotion_entry["gate"] = gate_report
            gate_report_path = (
                self.run_dir / "reports" / f"gate_promoted_{self.generation:04d}.json"
            )
            _atomic_json(gate_report_path, gate_report)
            if gate_report["passed"]:
                release = BENCH_ROOT / "releases" / RELEASE_TAG
                self.league["status"] = "champion"
                self.league["release"] = str(release)
                self.league["gate_report"] = str(gate_report_path.relative_to(self.run_dir))
                release_request = snapshot, gate_report
            else:
                self.league["status"] = "candidate"
        report = {
            "generation": self.generation,
            "games": len(games),
            "positions": sum(len(game.positions) for game in games),
            "replay_size": min(len(self.replay) + len(staging), self.replay.capacity),
            "batch_size": batch_size,
            "candidate": str(candidate.relative_to(self.run_dir)),
            "promoted": promoted,
            "promotion": promotion.as_dict(),
            **train_stats,
        }
        _atomic_json(self.run_dir / "reports" / f"generation_{self.generation:04d}.json", report)
        pending = pending_replay_descriptor(staging, self.replay, self.run_dir)
        self._checkpoint(pending)
        self.replay.append_many(staging.records())
        shutil.rmtree(staging.directory)
        if release_request is not None:
            freeze_release(*release_request, self.config)
        return report

    def run(self, generations: int) -> list[dict]:
        reports = []
        try:
            for _ in range(generations):
                if self.league.get("status") == "champion":
                    break
                reports.append(self.run_generation())
        except ResourceLimitReached as exc:
            self._handoff(str(exc))
            raise
        return reports
