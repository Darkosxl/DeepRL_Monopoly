"""Native MCTS shards and canonical Gemma JSONL teacher data."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .adapters import SearchAdapter
from .config import BenchmarkConfig
from .contracts import ReplayPosition
from .engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, ROOT, SharedGame, legal_mask
from .model import MonopolyZeroNet
from .storage import MASK_BYTES, MAX_SPARSE_ACTIONS, _atomic_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slm_contract():
    directory = ROOT / "SLM_HANDMADE_MONOPOLY"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module("monopoly_qlora")


def _stratified(rows: list[dict], limit: int = 64) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["phase"], row["action_family"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row["step"])
    selected = []
    while buckets and len(selected) < limit:
        for key in sorted(tuple(buckets)):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == limit:
                break
    return selected


def collect_teacher_game(
    model: MonopolyZeroNet,
    model_hash: str,
    *,
    game_id: int,
    seed: int,
    config: BenchmarkConfig,
) -> tuple[list[ReplayPosition], list[dict], dict]:
    contract = _slm_contract()
    game = SharedGame.new(seed, config.max_rounds)
    policies = {
        seat: SearchAdapter(model, config.search, self_play=False) for seat in range(NUM_PLAYERS)
    }
    positions = []
    rows = []
    latencies = []
    for step in range(config.max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN):
        if game.env.done:
            break
        actor = game.env.whose_turn()
        legal = tuple(game.env.get_allowed_actions(actor))
        if len(legal) == 1:
            action = legal[0]
        else:
            result = policies[actor].choose_action(
                game,
                actor,
                seed * 1_000_003 + step * 17 + actor,
            )
            action = result.chosen_action
            if action not in legal:
                raise RuntimeError(f"Teacher selected illegal action {action}")
            latencies.append(result.latency_s)
            positions.append(
                ReplayPosition(
                    state=game.env._get_state(actor),
                    legal_mask=legal_mask(legal),
                    visits=result.visits,
                    q_values=result.q_values,
                    selected_action=action,
                    value=result.root_value,
                    actor_id=actor,
                    game_id=game_id,
                )
            )
            row = contract.make_dataset_row(
                env=game.env,
                actor_pid=actor,
                game_id=str(game_id),
                seed=seed,
                step=step,
                teacher_policy="mcts",
                teacher_candidates={
                    candidate: {"score": float(values[actor])}
                    for candidate, values in result.q_values.items()
                },
                teacher_action=action,
                behavior_action=action,
                exploratory=False,
                relabeled_action=action,
                outcome="pending",
                teacher_bundle_hash=model_hash,
            )
            row.update(
                {
                    "mcts_visits": {str(candidate): visits for candidate, visits in result.visits.items()},
                    "mcts_q_values": {str(candidate): list(values) for candidate, values in result.q_values.items()},
                    "root_value": list(result.root_value),
                    "simulations": result.simulations,
                    "latency_s": result.latency_s,
                }
            )
            rows.append(row)
        game.step(action)

    if not game.env.done:
        raise RuntimeError(f"Teacher game {game_id} did not complete")
    winner = game.env.winner()
    outcome = [0.0] * NUM_PLAYERS
    outcome[winner] = 1.0
    for position in positions:
        position.outcome = tuple(outcome)  # type: ignore[assignment]
    for row in rows:
        row["winner"] = winner
        row["outcome"] = "win" if row["actor_pid"] == winner else "loss"
    return positions, _stratified(rows), {
        "game_id": game_id,
        "seed": seed,
        "winner": winner,
        "positions": len(positions),
        "slm_positions": min(len(rows), 64),
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def _native_arrays(positions: list[ReplayPosition]) -> dict[str, np.ndarray]:
    count = len(positions)
    arrays = {
        "states": np.empty((count, 300), dtype=np.float32),
        "legal_masks": np.empty((count, MASK_BYTES), dtype=np.uint8),
        "actions": np.full((count, MAX_SPARSE_ACTIONS), -1, dtype=np.int32),
        "visit_counts": np.zeros((count, MAX_SPARSE_ACTIONS), dtype=np.int32),
        "q_values": np.zeros((count, MAX_SPARSE_ACTIONS, NUM_PLAYERS), dtype=np.float32),
        "lengths": np.zeros(count, dtype=np.uint8),
        "selected_actions": np.empty(count, dtype=np.int32),
        "values": np.empty((count, NUM_PLAYERS), dtype=np.float32),
        "outcomes": np.empty((count, NUM_PLAYERS), dtype=np.float32),
        "actors": np.empty(count, dtype=np.int8),
        "game_ids": np.empty(count, dtype=np.int64),
    }
    for index, position in enumerate(positions):
        actions = sorted(position.visits, key=lambda action: (-position.visits[action], action))
        if len(actions) > MAX_SPARSE_ACTIONS:
            raise ValueError("Native shard cannot encode more than 64 retained actions")
        length = len(actions)
        arrays["states"][index] = position.state
        arrays["legal_masks"][index] = np.packbits(position.legal_mask, bitorder="little")
        arrays["actions"][index, :length] = actions
        arrays["visit_counts"][index, :length] = [position.visits[action] for action in actions]
        arrays["q_values"][index, :length] = [position.q_values[action] for action in actions]
        arrays["lengths"][index] = length
        arrays["selected_actions"][index] = position.selected_action
        arrays["values"][index] = position.value
        arrays["outcomes"][index] = position.outcome
        arrays["actors"][index] = position.actor_id
        arrays["game_ids"][index] = position.game_id
    return arrays


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, path)


def export_teacher(
    champion: str | Path,
    *,
    games: int = 256,
    output_dir: str | Path,
    config: BenchmarkConfig | None = None,
    shard_positions: int = 4_096,
) -> dict:
    config = config or BenchmarkConfig()
    output = Path(output_dir)
    native = output / "native"
    slm = output / "slm"
    native.mkdir(parents=True, exist_ok=True)
    slm.mkdir(parents=True, exist_ok=True)
    model = MonopolyZeroNet.load_inference(champion)
    model_hash = file_sha256(champion)
    all_positions = []
    split_rows = {"train": [], "validation": [], "test": []}
    game_reports = []
    for game_index in range(games):
        positions, rows, report = collect_teacher_game(
            model,
            model_hash,
            game_id=game_index,
            seed=config.seeds.teacher + game_index,
            config=config,
        )
        all_positions.extend(positions)
        split = "validation" if game_index % 10 == 0 else "test" if game_index % 10 == 1 else "train"
        split_rows[split].extend(rows)
        game_reports.append(report)

    shards = []
    for start in range(0, len(all_positions), shard_positions):
        path = native / f"teacher_{start // shard_positions:05d}.npz"
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(temporary, **_native_arrays(all_positions[start : start + shard_positions]))
        os.replace(temporary, path)
        shards.append({"path": str(path.relative_to(output)), "sha256": file_sha256(path)})
    for split, rows in split_rows.items():
        _write_jsonl(slm / f"{split}.jsonl", rows)
    manifest = {
        "schema": 1,
        "ruleset": config.ruleset,
        "champion": str(Path(champion)),
        "champion_sha256": model_hash,
        "games": games,
        "native_positions": len(all_positions),
        "slm_rows": {name: len(rows) for name, rows in split_rows.items()},
        "whole_game_split": "game_id modulo 10: 0 validation, 1 test, else train",
        "shards": shards,
        "game_reports": game_reports,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest
