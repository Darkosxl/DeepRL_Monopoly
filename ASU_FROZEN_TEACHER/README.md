# ASU Frozen Teacher v1

`ASU_FROZEN_TEACHER` contains two deterministic, versioned policies for the
repository's `ppo-plus-v2` Monopoly simulator:

- `asu_value_v1`: one-step action evaluation with a four-part state value.
- `asu_rollout_v1`: an eight-by-eight strength-first truncated rollout whose
  subsequent seats all use `asu_value_v1` for 32 decisions.

Both are **ASU-inspired reconstructions**, not releases or reproductions of the
authors' agents. The value structure and five-turn/five-lap horizons come from
Gopalakrishnan et al.'s [ASU heuristic paper](https://arxiv.org/pdf/2107.04303).
The rollout shape is inspired by the later
[truncated-rollout work](https://arxiv.org/pdf/2302.14208), which describes short
rollouts followed by a state evaluator but does not publish this package's exact
budget, action rules, seeds, or `ppo-plus-v2` behavior. Those inferred choices
are frozen in `spec.py` and were not tuned on benchmark seeds.

## Public API

```python
from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1

agent = ASUValueV1(player_id=0)
action = agent.choose_action(env)
decision = agent.decide(env)

print(decision.frozen_spec_fingerprint)
print(decision.selected.value)
print(decision.safety_rejections)
```

`Decision`, `CandidateScore`, `ValueBreakdown`, `RentProjection`,
`SafetyBreakdown`, and `SafetyRejection` are frozen dataclasses. `decide()`
reports every legal action, all four value components, both safety margins,
trade gains or auction ceilings where applicable, rejection reasons, and the
canonical SHA-256 spec fingerprint. `Decision.to_dict()` is suitable for future
SLM labels.

The canonical specification is serialized with sorted JSON keys and compact
separators. `FROZEN_SPEC_CANONICAL_JSON`, `FROZEN_SPEC_HASH`, and
`FROZEN_SPEC_FINGERPRINT` are exported so datasets and evaluations can reject
labels produced by a different policy definition.

## Frozen reconstruction

The non-terminal value is

```text
V(s) = M_assets + R_short + R_long + M_monopoly
```

- `M_assets` is list price plus paid development cost for unmortgaged holdings;
  cash is intentionally excluded.
- `R_short` exhaustively enumerates both dice over each active player's next
  five complete turns. It accounts for extra rolls, triple doubles, all jail
  turns, Go To Jail, current railroad counts, and the actual dice total used by
  utilities.
- `R_long` uses five laps, `40/7` landing opportunities per lap, and a uniform
  distribution over 28 deeds.
- `M_monopoly` starts with `cash + 5*$200 + R_long`, buys missing group deeds at
  list price, optionally unmortgages held deeds, and maximizes resulting group
  rent under the current bank's house/hotel inventory and the simulator's
  uneven-building rules. It is divided by `2 ** missing_deeds`.

Discretionary spending must satisfy both frozen bankruptcy gates:

```text
cash_after + next_round_net_rent >= 200
cash_after + next_round_rent_income + liquidatable_worth
    - worst_reachable_rent > 0
```

Liquidatable worth follows the full simulator sequence: legal half-price
development sales followed by deed mortgage/sale value. Forced progress and
debt-liquidation actions are never pruned. Trades are scored as accepted
counterfactual transfers and require proposer gain `> 0`, recipient gain
`>= 0`, and both parties' safety gates. Auction bids use the largest legal
increment whose total bid is no greater than the deed's marginal ASU value and
passes safety; otherwise the agent passes.

The rollout policy uses streams seeded `0..7` for every shortlisted action, so
candidate actions receive action-independent common random numbers. A rollout
never mutates the caller's environment and restores Python, NumPy, and imported
Torch process RNG states.

## Seat-balanced evaluator

Each supplied seed creates a four-game paired block. The focus policy occupies
physical seats 0, 1, 2, and 3 once while the game seed is held fixed.

```bash
python -m ASU_FROZEN_TEACHER.evaluate \
  --focus asu-value-v1 \
  --opponents fixed-a fixed-b fixed-c \
  --seeds 0 1 --pretty
```

Available scripted IDs are `fixed-a` through `fixed-f`. Checkpoints require an
explicit path:

```bash
python -m ASU_FROZEN_TEACHER.evaluate \
  --focus ddqn:/absolute/path/ddqn.pt \
  --opponents ppo:/absolute/path/ppo.pt \
              cfr:/absolute/path/cfr.pkl.gz \
              asu-rollout-v1 \
  --seeds 10 --output artifacts/asu_eval.json
```

PPO and DDQN metadata must match ruleset `ppo-plus-v2`, state dimension 300,
action dimension 2958, and checkpoint format 3. Neural evaluation uses
evaluation mode, inference mode, legal-action masking, and deterministic
argmax. CFR checkpoints use pickle and must only come from a trusted source.
The repository's fixed policies sometimes return `END_TURN` in states where
only liquidation or trade actions are legal. Their evaluator adapter applies
the same `END_TURN`-otherwise-first-legal compatibility fallback used by the
existing training and statistics drivers, and reports every use in JSON. ASU
and checkpoint policies remain strictly checked for illegal output.

The JSON includes policy IDs, ruleset and frozen-spec hashes, checkpoint file
hashes, paired seeds, per-game winner/round/decision/net-worth records, Wilson
intervals, truncations, and elapsed time. Decision-cap truncations have no
winner and are excluded from win rates and Wilson intervals; their current
net-worth leader is reported separately as `provisional_leader`. Its disclaimer
is intentional:
`ppo-plus-v2` results from these reconstructed policies do **not** reproduce or
estimate win rates reported in either paper.
