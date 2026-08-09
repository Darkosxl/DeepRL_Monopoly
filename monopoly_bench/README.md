# MonopolyZero v1 benchmark

This directory is an isolated training and evaluation pipeline for this
repository's `ppo-plus-v2` simulator. It does not claim performance on official
Monopoly or against professional human play.

Public commands:

```bash
python -m monopoly_bench smoke
python -m monopoly_bench train --run-dir monopoly_bench/runs/example \
  --bootstrap-ppo artifacts/ppo_plus/ppo_hybrid_2000_v2.pt --fallback-colab
python -m monopoly_bench gate --run-dir monopoly_bench/runs/example
python -m monopoly_bench evaluate --champion path/to/incumbent.pt \
  --candidate path/to/candidate.pt
python -m monopoly_bench export-teacher --champion path/to/champion.pt --games 256
```

The defaults in `configs/v1.json` are frozen. A model remains a `candidate`
until every available fixed, PPO-v2, and CFR-v2 matchup passes the full gate.
Only that successful gate can create the immutable local release bundle.

