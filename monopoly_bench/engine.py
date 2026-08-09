"""Read-only bridge to the ppo-plus-v2 simulator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path
import random
from types import MethodType
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from monopoly_game_engine.actions import ACTION_SPACE_SIZE, OFFSETS, ActionType  # noqa: E402
from monopoly_game_engine.constants import (  # noqa: E402
    GO_SALARY,
    JAIL_BAIL,
    JAIL_SQUARE,
    MAX_JAIL_TURNS,
    NUM_PLAYERS,
    RULESET_VERSION,
)
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402


if (RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE, NUM_PLAYERS) != (
    "ppo-plus-v2",
    300,
    2_958,
    4,
):
    raise RuntimeError("The installed Monopoly engine is not ppo-plus-v2")


DICE_OUTCOMES = tuple(product(range(1, 7), repeat=2))
MAX_DECISIONS_PER_TURN = 256


@dataclass(slots=True)
class SharedGame:
    """A cloneable engine state with private stdlib RNG state."""

    env: MonopolyEnv
    random_state: object

    @classmethod
    def new(cls, seed: int, max_rounds: int = 200) -> "SharedGame":
        outer = random.getstate()
        try:
            random.seed(seed)
            env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
            state = random.getstate()
        finally:
            random.setstate(outer)
        return cls(env, state)

    def clone(self) -> "SharedGame":
        return copy.deepcopy(self)

    def step(self, action: int):
        outer = random.getstate()
        try:
            random.setstate(self.random_state)
            result = self.env.step(action)
            self.random_state = random.getstate()
            return result
        finally:
            random.setstate(outer)


def unwrap(game: MonopolyEnv | SharedGame) -> MonopolyEnv:
    return game.env if isinstance(game, SharedGame) else game


def clone_env(game: MonopolyEnv | SharedGame) -> MonopolyEnv:
    return copy.deepcopy(unwrap(game))


def actor_order(actor_id: int) -> tuple[int, ...]:
    if not 0 <= actor_id < NUM_PLAYERS:
        raise ValueError(f"Invalid actor id: {actor_id}")
    return (actor_id, *(pid for pid in range(NUM_PLAYERS) if pid != actor_id))


def relative_to_physical(values: Iterable[float], actor_id: int) -> np.ndarray:
    values = np.asarray(tuple(values), dtype=np.float32)
    if values.shape != (NUM_PLAYERS,):
        raise ValueError(f"Expected {NUM_PLAYERS} values, got {values.shape}")
    physical = np.empty(NUM_PLAYERS, dtype=np.float32)
    physical[list(actor_order(actor_id))] = values
    return physical


def physical_to_relative(values: Iterable[float], actor_id: int) -> np.ndarray:
    values = np.asarray(tuple(values), dtype=np.float32)
    if values.shape != (NUM_PLAYERS,):
        raise ValueError(f"Expected {NUM_PLAYERS} values, got {values.shape}")
    return values[list(actor_order(actor_id))]


def terminal_value(env: MonopolyEnv) -> np.ndarray:
    if not env.done:
        raise ValueError("Terminal backup requested for a live game")
    value = np.zeros(NUM_PLAYERS, dtype=np.float32)
    winner = env.winner()
    if winner is not None:
        value[winner] = 1.0
    return value


def action_family(action: int) -> str:
    names = sorted(OFFSETS, key=OFFSETS.get)
    for index, name in enumerate(names):
        end = OFFSETS[names[index + 1]] if index + 1 < len(names) else ACTION_SPACE_SIZE
        if OFFSETS[name] <= action < end:
            return name
    raise ValueError(f"Action outside ppo-plus-v2 space: {action}")


def legal_mask(actions: Iterable[int]) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    actions = tuple(int(action) for action in actions)
    if not actions:
        raise ValueError("A decision must have at least one legal action")
    mask[list(actions)] = True
    return mask


def _forced_roll(self: MonopolyEnv, pid: int, info: dict, dice: tuple[int, int]) -> None:
    """Engine-equivalent roll with an explicit ordered outcome."""
    d1, d2 = dice
    if (d1, d2) not in DICE_OUTCOMES:
        raise ValueError(f"Invalid ordered dice outcome: {dice}")
    player = self.players[pid]
    self.last_dice = (d1, d2)
    info["dice"] = (d1, d2)
    self.has_rolled = True

    was_in_jail = player.in_jail
    if was_in_jail:
        player.jail_turns += 1
        if d1 == d2:
            player.in_jail = False
            player.jail_turns = 0
        elif player.jail_turns >= MAX_JAIL_TURNS:
            player.cash -= min(JAIL_BAIL, player.cash)
            player.in_jail = False
            player.jail_turns = 0
        else:
            return

    if not was_in_jail and d1 == d2:
        self.consecutive_doubles += 1
        if self.consecutive_doubles >= 3:
            player.position = JAIL_SQUARE
            player.in_jail = True
            player.jail_turns = 0
            self.consecutive_doubles = 0
            self.extra_roll_pending = False
            info["three_doubles"] = True
            return
        self.extra_roll_pending = True
    else:
        self.consecutive_doubles = 0
        self.extra_roll_pending = False

    new_position = (player.position + d1 + d2) % 40
    if new_position < player.position and not player.in_jail:
        player.cash += GO_SALARY
    player.position = new_position
    self._handle_landing(pid, d1 + d2, info)


def transition(
    game: MonopolyEnv | SharedGame,
    action: int,
    dice: tuple[int, int] | None = None,
) -> MonopolyEnv:
    """Clone and apply one action without consuming ambient RNG state."""
    env = clone_env(game)
    if action == int(ActionType.ROLL_DICE):
        if dice is None:
            raise ValueError("ROLL_DICE requires an explicit chance outcome")
        env._do_roll = MethodType(lambda self, pid, info: _forced_roll(self, pid, info, dice), env)
        try:
            env.step(action)
        finally:
            del env.__dict__["_do_roll"]
    else:
        if dice is not None:
            raise ValueError("Dice may only accompany ROLL_DICE")
        env.step(action)
    return env


def engine_hashes() -> dict[str, str]:
    files = ("actions.py", "constants.py", "env.py", "state.py", "networks.py", "agents_fixed.py")
    directory = ROOT / "monopoly_game_engine"
    return {
        str((directory / name).relative_to(ROOT)): hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in files
    }


def source_hashes() -> dict[str, str]:
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *(ROOT / "ASU_FROZEN_TEACHER").glob("*.py"),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
