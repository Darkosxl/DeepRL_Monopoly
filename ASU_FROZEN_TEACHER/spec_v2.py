"""Documentation-only specification for the v2 ASU teachers.

Unlike ``ASU_FROZEN_TEACHER/spec.py``, nothing in this repository gates on
``SPEC_V2_HASH`` -- no training or release-gate code checks it. It exists
purely so two differently-tuned versions of v2's search constants can be
told apart in an exported ``Decision.to_dict()`` record, and is expected to
change whenever the tunable constants below change (see the win-rate tuning
loop in the implementation plan). ``core_v2.py``'s constants are the source
of truth; this dict is descriptive metadata, mirroring how ``spec.py``'s
``_SPEC`` independently documents ``core.py``'s frozen constants.
"""

from __future__ import annotations

import hashlib
import json


ASU_VALUE_V2 = "asu_value_v2"
ASU_ROLLOUT_V2 = "asu_rollout_v2"

_SPEC_V2 = {
    "schema": "asu-frozen-teacher/v2-tunable",
    "ruleset": "ppo-plus-v2",
    "status": "ASU-inspired reconstruction v2: same V(s) semantics as v1, faster/smarter search",
    "policies": {
        ASU_VALUE_V2: "one-step value policy with fast cloning, bucketed dice, and a trade pre-filter",
        ASU_ROLLOUT_V2: "successive-halving rollout policy over ASUValueV2",
    },
    "clone": "hand-written field-level clone instead of copy.deepcopy",
    "dice": "36 ordered pairs deduplicated to 15 (total, is_double) buckets, weighted average",
    "trade_prefilter": {
        "shortlist": 24,
        "score": "asset price delta + hypothetical-group-rent monopoly delta - cash floor penalty",
    },
    "rollout": {
        "initial_shortlist": 12,
        "allocator": "successive halving",
        "initial_rollouts_per_candidate": 4,
        "round_budget_multiplier": 2,
        "elimination_z": 1.0,
        "max_total_rollouts": 128,
    },
}

FROZEN_SPEC_V2_CANONICAL_JSON = json.dumps(
    _SPEC_V2, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
SPEC_V2_HASH = hashlib.sha256(FROZEN_SPEC_V2_CANONICAL_JSON.encode("ascii")).hexdigest()
SPEC_V2_FINGERPRINT = f"sha256:{SPEC_V2_HASH}"


__all__ = [
    "ASU_ROLLOUT_V2",
    "ASU_VALUE_V2",
    "FROZEN_SPEC_V2_CANONICAL_JSON",
    "SPEC_V2_FINGERPRINT",
    "SPEC_V2_HASH",
]
