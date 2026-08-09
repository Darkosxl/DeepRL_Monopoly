"""Immutable public records emitted by the frozen teachers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValueBreakdown:
    m_assets: float
    r_short: float
    r_long: float
    m_monopoly: float
    terminal_utility: float
    total: float

    @property
    def assets(self) -> float:
        return self.m_assets

    @property
    def short_rent(self) -> float:
        return self.r_short

    @property
    def long_rent(self) -> float:
        return self.r_long

    @property
    def monopoly(self) -> float:
        return self.m_monopoly


@dataclass(frozen=True, slots=True)
class RentProjection:
    income: float
    exposure: float
    net: float
    worst_reachable_rent: float


@dataclass(frozen=True, slots=True)
class SafetyBreakdown:
    cash_after: float
    next_round_net_rent: float
    next_round_rent_income: float
    liquidatable_worth: float
    worst_reachable_rent: float
    cash_floor_margin: float
    solvency_margin: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SafetyRejection:
    action: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    action: int
    description: str
    value: ValueBreakdown
    safety: SafetyBreakdown
    eligible: bool
    mandatory: bool
    forced: bool
    semantic_priority: int
    rejection_reasons: tuple[str, ...] = ()
    proposer_gain: float | None = None
    recipient_gain: float | None = None
    auction_ceiling: float | None = None
    shortlisted: bool = False
    rollout_scores: tuple[float, ...] = ()
    rollout_mean: float | None = None

    @property
    def score(self) -> float:
        return self.value.total if self.rollout_mean is None else self.rollout_mean


@dataclass(frozen=True, slots=True)
class Decision:
    policy_id: str
    player_id: int
    selected_action: int
    candidates: tuple[CandidateScore, ...]
    safety_rejections: tuple[SafetyRejection, ...]
    frozen_spec_hash: str
    frozen_spec_fingerprint: str
    rollout_seeds: tuple[int, ...] = ()

    @property
    def action(self) -> int:
        return self.selected_action

    @property
    def selected(self) -> CandidateScore:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.action == self.selected_action
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready labeling record."""
        return asdict(self)


__all__ = [
    "CandidateScore",
    "Decision",
    "RentProjection",
    "SafetyBreakdown",
    "SafetyRejection",
    "ValueBreakdown",
]
