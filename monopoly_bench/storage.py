"""Memory-mapped replay and atomic deterministic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import hashlib
from typing import Iterable

import numpy as np
import torch

from .config import BenchmarkConfig
from .contracts import ReplayPosition
from .engine import (
    ACTION_SPACE_SIZE,
    NUM_PLAYERS,
    RULESET_VERSION,
    STATE_DIM,
    engine_hashes,
    source_hashes,
)


MASK_BYTES = (ACTION_SPACE_SIZE + 7) // 8
MAX_SPARSE_ACTIONS = 64
CHECKPOINT_VERSION = 1


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ReplayBuffer:
    """A circular 100k-position buffer with fixed sparse MCTS slots."""

    _LAYOUT = {
        "states": (np.float32, (STATE_DIM,)),
        "masks": (np.uint8, (MASK_BYTES,)),
        "actions": (np.int32, (MAX_SPARSE_ACTIONS,)),
        "visit_counts": (np.int32, (MAX_SPARSE_ACTIONS,)),
        "q_values": (np.float32, (MAX_SPARSE_ACTIONS, NUM_PLAYERS)),
        "lengths": (np.uint8, ()),
        "selected": (np.int32, ()),
        "values": (np.float32, (NUM_PLAYERS,)),
        "outcomes": (np.float32, (NUM_PLAYERS,)),
        "actors": (np.int8, ()),
        "game_ids": (np.int64, ()),
    }

    def __init__(self, directory: str | Path, capacity: int = 100_000, *, create: bool = False):
        self.directory = Path(directory)
        self.metadata_path = self.directory / "metadata.json"
        if create:
            if capacity < 1:
                raise ValueError("Replay capacity must be positive")
            self.directory.mkdir(parents=True, exist_ok=True)
            self.capacity, self.cursor, self.size, self.total = capacity, 0, 0, 0
            mode = "w+"
        else:
            metadata = json.loads(self.metadata_path.read_text())
            expected = (RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE, MAX_SPARSE_ACTIONS)
            actual = tuple(metadata.get(name) for name in ("ruleset", "state_dim", "action_dim", "max_sparse"))
            if actual != expected:
                raise ValueError(f"Incompatible replay metadata: {actual}; expected {expected}")
            self.capacity = int(metadata["capacity"])
            self.cursor = int(metadata["cursor"])
            self.size = int(metadata["size"])
            self.total = int(metadata["total"])
            mode = "r+"

        self.arrays: dict[str, np.memmap] = {}
        for name, (dtype, tail) in self._LAYOUT.items():
            self.arrays[name] = np.memmap(
                self.directory / f"{name}.mmap",
                dtype=dtype,
                mode=mode,
                shape=(self.capacity, *tail),
            )
        if create:
            self.arrays["actions"][:] = -1
            self.arrays["selected"][:] = -1
            self.arrays["game_ids"][:] = -1
            self.flush()

    def __len__(self) -> int:
        return self.size

    def _metadata(self) -> dict:
        return {
            "format_version": 1,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "max_sparse": MAX_SPARSE_ACTIONS,
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "total": self.total,
        }

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()
        _atomic_json(self.metadata_path, self._metadata())

    def append(self, position: ReplayPosition) -> int:
        return self.append_many([position])[0]

    def append_many(self, positions: Iterable[ReplayPosition]) -> list[int]:
        indices = []
        for position in positions:
            index = self.cursor
            self._write(index, position)
            indices.append(index)
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.total += 1
        if indices:
            self.flush()
        return indices

    def _write(self, index: int, position: ReplayPosition) -> None:
        state = np.asarray(position.state, dtype=np.float32)
        mask = np.asarray(position.legal_mask, dtype=np.bool_)
        if state.shape != (STATE_DIM,) or mask.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(f"Bad replay shapes: state={state.shape}, mask={mask.shape}")
        if not np.isfinite(state).all():
            raise ValueError("Replay state contains a non-finite value")
        if not 0 <= position.selected_action < ACTION_SPACE_SIZE or not mask[position.selected_action]:
            raise ValueError("Selected replay action is not legal")

        actions = sorted(position.visits, key=lambda action: (-position.visits[action], action))
        if len(actions) > MAX_SPARSE_ACTIONS:
            raise ValueError(f"Search retained {len(actions)} actions; replay supports {MAX_SPARSE_ACTIONS}")
        unknown_q = set(actions) - set(position.q_values)
        if unknown_q:
            raise ValueError(f"Replay is missing Q vectors for actions: {sorted(unknown_q)[:3]}")

        self.arrays["states"][index] = state
        packed = np.packbits(mask, bitorder="little")
        self.arrays["masks"][index] = packed
        self.arrays["actions"][index] = -1
        self.arrays["visit_counts"][index] = 0
        self.arrays["q_values"][index] = 0
        length = len(actions)
        if length:
            self.arrays["actions"][index, :length] = actions
            self.arrays["visit_counts"][index, :length] = [position.visits[action] for action in actions]
            self.arrays["q_values"][index, :length] = [position.q_values[action] for action in actions]
        self.arrays["lengths"][index] = length
        self.arrays["selected"][index] = position.selected_action
        self.arrays["values"][index] = position.value
        self.arrays["outcomes"][index] = position.outcome
        self.arrays["actors"][index] = position.actor_id
        self.arrays["game_ids"][index] = position.game_id

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError("Cannot sample an empty replay")
        count = min(int(batch_size), self.size)
        indices = rng.choice(self.size, size=count, replace=self.size < count)
        result = {name: np.asarray(array[indices]).copy() for name, array in self.arrays.items()}
        result["legal_masks"] = np.unpackbits(result.pop("masks"), axis=1, bitorder="little")[:, :ACTION_SPACE_SIZE].astype(np.bool_)
        result["indices"] = indices
        return result

    def records(self, *, game_id: int | None = None) -> list[ReplayPosition]:
        indices = range(self.size)
        if game_id is not None:
            indices = [index for index in indices if int(self.arrays["game_ids"][index]) == game_id]
        records = []
        for index in indices:
            length = int(self.arrays["lengths"][index])
            actions = [int(action) for action in self.arrays["actions"][index, :length]]
            mask = np.unpackbits(self.arrays["masks"][index], bitorder="little")[:ACTION_SPACE_SIZE].astype(np.bool_)
            records.append(
                ReplayPosition(
                    state=np.asarray(self.arrays["states"][index]).copy(),
                    legal_mask=mask,
                    visits={action: int(self.arrays["visit_counts"][index, slot]) for slot, action in enumerate(actions)},
                    q_values={action: tuple(float(value) for value in self.arrays["q_values"][index, slot]) for slot, action in enumerate(actions)},
                    selected_action=int(self.arrays["selected"][index]),
                    value=tuple(float(value) for value in self.arrays["values"][index]),
                    outcome=tuple(float(value) for value in self.arrays["outcomes"][index]),
                    actor_id=int(self.arrays["actors"][index]),
                    game_id=int(self.arrays["game_ids"][index]),
                )
            )
        return records


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True, slots=True)
class ResumeState:
    generation: int
    replay_cursor: int
    replay_size: int
    league: dict
    promotion_history: list[dict]
    pending_replay: dict | None = None


def replay_state(replay: ReplayBuffer) -> tuple[int, int, int]:
    return replay.cursor, replay.size, replay.total


def replay_directory_hash(directory: str | Path) -> str:
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def pending_replay_descriptor(
    staging: ReplayBuffer,
    replay: ReplayBuffer,
    run_dir: str | Path,
) -> dict:
    count = len(staging)
    if count < 1:
        raise ValueError("A pending replay transaction cannot be empty")
    before = replay_state(replay)
    after = (
        (replay.cursor + count) % replay.capacity,
        min(replay.size + count, replay.capacity),
        replay.total + count,
    )
    return {
        "path": str(staging.directory.relative_to(Path(run_dir))),
        "count": count,
        "before": list(before),
        "after": list(after),
        "sha256": replay_directory_hash(staging.directory),
    }


class CheckpointManager:
    @staticmethod
    def save(
        path: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        config: BenchmarkConfig,
        replay: ReplayBuffer,
        generation: int,
        league: dict,
        promotion_history: list[dict],
        pending_replay: dict | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": CHECKPOINT_VERSION,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "config": config.as_dict(),
                "engine_hashes": engine_hashes(),
                "source_hashes": source_hashes(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "replay_cursor": replay.cursor,
                "replay_size": replay.size,
                "replay_total": replay.total,
                "generation": generation,
                "league": league,
                "promotion_history": promotion_history,
                "pending_replay": pending_replay,
                "rng": capture_rng_state(),
            },
            temporary,
        )
        os.replace(temporary, destination)

    @staticmethod
    def load(
        path: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        config: BenchmarkConfig,
        replay: ReplayBuffer,
        strict_source: bool = True,
    ) -> ResumeState:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = (CHECKPOINT_VERSION, RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE)
        actual = tuple(payload.get(name) for name in ("format_version", "ruleset", "state_dim", "action_dim"))
        if actual != expected:
            raise ValueError(f"Incompatible checkpoint metadata: {actual}; expected {expected}")
        if payload.get("config") != config.as_dict():
            raise ValueError("Checkpoint configuration differs from this run")
        if payload.get("engine_hashes") != engine_hashes():
            raise ValueError("Checkpoint rejected because the ppo-plus-v2 engine drifted")
        if strict_source and payload.get("source_hashes") != source_hashes():
            raise ValueError("Checkpoint rejected because benchmark source drifted")
        pending = payload.get("pending_replay")
        expected_replay = (
            tuple(int(value) for value in pending["after"])
            if pending is not None
            else (
                int(payload.get("replay_cursor", -1)),
                int(payload.get("replay_size", -1)),
                int(payload.get("replay_total", -1)),
            )
        )
        if (
            int(payload.get("replay_cursor", -1)),
            int(payload.get("replay_size", -1)),
            int(payload.get("replay_total", -1)),
        ) != expected_replay and pending is None:
            raise ValueError("Checkpoint replay metadata is internally inconsistent")
        if replay_state(replay) != expected_replay:
            raise ValueError("Checkpoint replay cursor does not match the memmap")

        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        device = next(model.parameters()).device
        for optimizer_state in optimizer.state.values():
            for name, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[name] = value.to(device)
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng"])
        return ResumeState(
            generation=int(payload["generation"]),
            replay_cursor=int(payload["replay_cursor"]),
            replay_size=int(payload["replay_size"]),
            league=dict(payload["league"]),
            promotion_history=list(payload["promotion_history"]),
            pending_replay=None if pending is None else dict(pending),
        )

    @staticmethod
    def reconcile_replay(
        path: str | Path,
        replay: ReplayBuffer,
        run_dir: str | Path,
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pending = payload.get("pending_replay")
        if pending is None:
            return
        before = tuple(int(value) for value in pending["before"])
        after = tuple(int(value) for value in pending["after"])
        current = replay_state(replay)
        if current == after:
            return
        if current != before:
            raise ValueError(
                f"Replay transaction is neither before nor after commit: {current}"
            )
        staging_path = Path(run_dir) / pending["path"]
        if not staging_path.is_dir():
            raise ValueError("Pending replay journal is unavailable")
        staging = ReplayBuffer(staging_path)
        if len(staging) != int(pending["count"]):
            raise ValueError("Pending replay journal count changed")
        if replay_directory_hash(staging_path) != pending["sha256"]:
            raise ValueError("Pending replay journal hash changed")
        replay.append_many(staging.records())
        if replay_state(replay) != after:
            raise ValueError("Pending replay transaction committed to an unexpected cursor")
