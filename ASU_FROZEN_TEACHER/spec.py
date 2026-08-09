"""Canonical, untunable specification for the frozen ASU teachers."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType


ASU_VALUE_V1 = "asu_value_v1"
ASU_ROLLOUT_V1 = "asu_rollout_v1"

_SPEC = {
    "schema": "asu-frozen-teacher/v1",
    "ruleset": "ppo-plus-v2",
    "status": "ASU-inspired reconstruction; not an original-agent reproduction",
    "policies": {
        ASU_VALUE_V1: "deterministic one-step state-value policy",
        ASU_ROLLOUT_V1: "strength-first truncated-rollout policy",
    },
    "value": {
        "formula": "M_assets + R_short + R_long + M_monopoly",
        "assets": "list price plus paid development cost for unmortgaged deeds; cash excluded",
        "short_turns": 5,
        "short_projection": "exact 2d6 complete-turn enumeration including doubles, triple doubles, jail, Go To Jail, railroad counts, and utility dice totals",
        "long_laps": 5,
        "long_landing_opportunities_per_lap": "40/7",
        "long_distribution": "uniform across 28 deeds",
        "monopoly_discount": "2 ** missing_properties",
        "monopoly_funds": "cash + 5 * 200 + R_long",
        "monopoly_plan": "buy missing deeds at list price, optionally unmortgage current deeds, then maximize group rent under current ppo-plus-v2 house/hotel inventory and building rules",
        "terminal_utility": 1_000_000,
    },
    "safety": {
        "minimum_cash": 200,
        "gate_1": "cash_after + next_round_net_rent >= 200",
        "gate_2": "cash_after + next_round_rent_income + liquidatable_worth - worst_reachable_rent > 0",
        "liquidation": "full ppo-plus-v2 half-price development sales followed by mortgage/sale value",
        "forced_and_debt_liquidation_are_never_pruned": True,
    },
    "trade": {
        "evaluation": "counterfactual accepted transfer",
        "proposer_gain": "> 0",
        "recipient_gain": ">= 0",
        "both_parties_must_pass_safety": True,
    },
    "auction": {
        "ceiling": "nonnegative marginal ASU value of acquiring the deed",
        "choice": "largest legal increment whose total bid is at or below the ceiling and passes both safety gates; otherwise pass",
    },
    "tie_break": {
        "first": "frozen semantic action-family priority",
        "second": "lower action id",
    },
    "rollout": {
        "seed": 0,
        "shortlist": 8,
        "rollouts_per_action": 8,
        "subsequent_decisions": 32,
        "policy_for_all_seats": ASU_VALUE_V1,
        "randomness": "action-independent common-random-number streams",
        "leaf": "root ASU value; terminal states short-circuit",
    },
    "provenance": {
        "heuristic_paper": "https://arxiv.org/pdf/2107.04303",
        "truncated_rollout_follow_up": "https://arxiv.org/pdf/2302.14208",
        "inferred_not_tuned": [
            "ppo-plus-v2 action semantics",
            "cash minimum",
            "full liquidation accounting",
            "trade thresholds",
            "auction ceiling mechanics",
            "semantic tie priorities",
            "terminal utility",
            "rollout budget and seeds",
        ],
    },
}

FROZEN_SPEC_CANONICAL_JSON = json.dumps(
    _SPEC, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
FROZEN_SPEC_HASH = hashlib.sha256(
    FROZEN_SPEC_CANONICAL_JSON.encode("ascii")
).hexdigest()
FROZEN_SPEC_FINGERPRINT = f"sha256:{FROZEN_SPEC_HASH}"


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


FROZEN_SPEC = _freeze(_SPEC)


__all__ = [
    "ASU_ROLLOUT_V1",
    "ASU_VALUE_V1",
    "FROZEN_SPEC",
    "FROZEN_SPEC_CANONICAL_JSON",
    "FROZEN_SPEC_FINGERPRINT",
    "FROZEN_SPEC_HASH",
]
