"""Small public data contracts shared by search, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


Vector = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    chosen_action: int
    visits: Mapping[int, int]
    q_values: Mapping[int, Vector]
    root_value: Vector
    simulations: int
    latency_s: float

    def as_dict(self) -> dict:
        return {
            "chosen_action": self.chosen_action,
            "visits": {str(k): int(v) for k, v in self.visits.items()},
            "q_values": {str(k): list(v) for k, v in self.q_values.items()},
            "root_value": list(self.root_value),
            "simulations": self.simulations,
            "latency_s": self.latency_s,
        }


@dataclass(slots=True)
class ReplayPosition:
    state: np.ndarray
    legal_mask: np.ndarray
    visits: Mapping[int, int]
    q_values: Mapping[int, Vector]
    selected_action: int
    value: Vector
    outcome: Vector = (0.0, 0.0, 0.0, 0.0)
    actor_id: int = 0
    game_id: int = -1


@dataclass(slots=True)
class GameResult:
    game_id: int
    seed: int
    winner: int | None
    completed: bool
    decisions: int
    positions: list[ReplayPosition] = field(default_factory=list)
    illegal_actions: int = 0
    fallbacks: int = 0
    crashes: int = 0
    search_latencies: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MatchSummary:
    games: int
    completed: int
    wins: int
    losses: int
    win_rate: float
    wilson_lower: float
    illegal_actions: int
    fallbacks: int
    crashes: int
    latency_p95_s: float

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
