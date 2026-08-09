"""Frozen v1 defaults and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    max_depth: int = 256
    max_width: int = 64
    widening_base: int = 8
    widening_scale: float = 4.0
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3
    temperature_rounds: int = 40
    visit_temperature: float = 1.0

    def __post_init__(self) -> None:
        if min(self.simulations, self.max_depth, self.max_width) < 1:
            raise ValueError("Search counts must be positive")
        if self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("PUCT and Dirichlet constants must be positive")
        if self.visit_temperature <= 0:
            raise ValueError("Visit temperature must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("Dirichlet epsilon must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    bootstrap_games: int = 256
    value_updates: int = 2_000
    games_per_generation: int = 32
    workers: int = 4
    replay_capacity: int = 100_000
    updates_per_generation: int = 1_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    promoted_snapshot_limit: int = 8
    promotion_games: int = 120
    promotion_win_rate: float = 0.55
    promotion_wilson_lower: float = 0.45
    expert_fraction: float = 0.20
    expert_decay_generations: int = 8
    asu_rollout_bootstrap_positions: int = 64

    def __post_init__(self) -> None:
        counts = (
            self.bootstrap_games,
            self.value_updates,
            self.games_per_generation,
            self.workers,
            self.replay_capacity,
            self.updates_per_generation,
            self.batch_size,
            self.expert_decay_generations,
        )
        if min(counts) < 1:
            raise ValueError("Training counts must be positive")
        if self.asu_rollout_bootstrap_positions < 0:
            raise ValueError("ASU rollout bootstrap positions cannot be negative")
        if not 0 <= self.expert_fraction <= 1:
            raise ValueError("Expert fraction must be in [0, 1]")
        if self.games_per_generation % self.workers:
            raise ValueError("Games per generation must divide evenly across workers")
        if self.games_per_generation % 4:
            raise ValueError("Population fractions require a multiple of four games")
        if self.promotion_games % 6:
            raise ValueError("Two-versus-two promotion games must balance all six seat pairs")


@dataclass(frozen=True, slots=True)
class GateConfig:
    screen_games: int = 20
    screen_win_rate: float = 0.45
    full_games: int = 100
    min_win_rate: float = 0.50
    min_wilson_lower: float = 0.40
    max_latency_p95_s: float = 2.0

    def __post_init__(self) -> None:
        if min(self.screen_games, self.full_games) < 1:
            raise ValueError("Gate game counts must be positive")
        if self.screen_games % 4 or self.full_games % 4:
            raise ValueError("Gate games must balance all four champion seats")
        if not all(0 <= value <= 1 for value in (self.screen_win_rate, self.min_win_rate, self.min_wilson_lower)):
            raise ValueError("Gate rates must be in [0, 1]")
        if self.max_latency_p95_s <= 0:
            raise ValueError("Gate latency threshold must be positive")


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    soft_rss_gib: float = 3.0
    hard_rss_gib: float = 4.0
    min_available_ram_gib: float = 2.0
    reserved_vram_gib: float = 1.0
    min_free_disk_gib: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.soft_rss_gib,
            self.hard_rss_gib,
            self.min_available_ram_gib,
            self.reserved_vram_gib,
            self.min_free_disk_gib,
        )
        if min(values) < 0 or self.hard_rss_gib <= self.soft_rss_gib:
            raise ValueError("Resource limits are invalid")


@dataclass(frozen=True, slots=True)
class SeedRanges:
    """Disjoint one-million-seed namespaces."""

    bootstrap: int = 100_000
    self_play: int = 1_100_000
    promotion: int = 2_100_000
    ladder: int = 3_100_000
    teacher: int = 4_100_000
    gate: int = 9_100_000
    span: int = 1_000_000

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError("Seed span must be positive")
        starts = sorted(
            (self.bootstrap, self.self_play, self.promotion, self.ladder, self.teacher, self.gate)
        )
        if any(right - left < self.span for left, right in zip(starts, starts[1:])):
            raise ValueError("Benchmark seed ranges overlap")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    ruleset: str = "ppo-plus-v2"
    state_dim: int = 300
    action_dim: int = 2_958
    players: int = 4
    max_rounds: int = 200
    search: SearchConfig = field(default_factory=SearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    seeds: SeedRanges = field(default_factory=SeedRanges)

    def __post_init__(self) -> None:
        if (self.ruleset, self.state_dim, self.action_dim, self.players) != (
            "ppo-plus-v2",
            300,
            2_958,
            4,
        ):
            raise ValueError("MonopolyZero v1 is frozen to ppo-plus-v2 dimensions")

    def as_dict(self) -> dict:
        return asdict(self)

    def assert_frozen_v1(self) -> None:
        if self != type(self)():
            raise ValueError("Champion gates require the exact frozen MonopolyZero v1 configuration")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, value: dict) -> "BenchmarkConfig":
        value = dict(value)
        for name, kind in (
            ("search", SearchConfig),
            ("training", TrainingConfig),
            ("gate", GateConfig),
            ("resources", ResourceConfig),
            ("seeds", SeedRanges),
        ):
            value[name] = kind(**value.get(name, {}))
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
