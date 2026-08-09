# MonopolyZero v1 benchmark

This directory is an isolated training and evaluation pipeline for this
repository's `ppo-plus-v2` simulator. It does not claim performance on official
Monopoly or against professional human play.

Public commands:

```bash
python -m monopoly_bench smoke
python -m monopoly_bench collect-asu --output path/to/asu-shard.npz \
  --games 32 --seed-base 100000 --rollout-positions 8
python -m monopoly_bench train --run-dir monopoly_bench/runs/example \
  --bootstrap-ppo artifacts/ppo_plus/ppo_hybrid_2000_v2.pt \
  --asu-expert-data path/to/asu-shard.npz --fallback-colab
python -m monopoly_bench gate --run-dir monopoly_bench/runs/example
python -m monopoly_bench evaluate --champion path/to/incumbent.pt \
  --candidate path/to/candidate.pt
python -m monopoly_bench export-teacher --champion path/to/champion.pt --games 256
```

The defaults in `configs/v1.json` are frozen. A model remains a `candidate`
until every available fixed, ASU-value, PPO-v2, CFR-v2, and mixed ASU/Deal
Maker/Gambler matchup passes the full gate, plus the ASU-rollout screen. Only
that successful gate can create the immutable local release bundle.

PPO supplies compatible initial actor weights only. Frozen ASU actions and real
game winners train the bootstrap policy/value network. Later generations learn
primarily from Max-N PUCT visits and winners, reserve four population games for
three-copy ASU-value opposition, and decay ASU imitation over eight generations.
The final inference checkpoint has no runtime dependency on either ASU policy.
