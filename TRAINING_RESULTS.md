# PPO-plus training results

Measured on 2026-08-09 with an NVIDIA GeForce RTX 4050 Laptop GPU. Generated
checkpoints are under `artifacts/`, which is intentionally ignored by Git.

## Hybrid PPO: 2,000 games

- Result: all 2,000 games completed.
- Wall time: 3,248.880 seconds (54m08.9s).
- Mean wall time: 1.62444 seconds/game.
- Peak process RSS: 1.60047 GiB.
- Peak CUDA allocation: 0.04535 GiB.
- Final 40-game win-rate window: 57.5%.
- Best 40-game win-rate window: 75.0%.
- Learned policy steps: 3,728,408.
- Checkpoint: `artifacts/ppo_plus/ppo_hybrid_2000.pt` (13.5 MiB).
- History: `artifacts/ppo_plus/ppo_hybrid_2000_history.json` (4.7 KiB).

The final model loaded successfully on CPU. A one-game inference smoke test
also completed; one game is not a statistically useful evaluation sample.

## Four-player Monte Carlo CFR: one full game

- Result: completed at the configured 200-round cap without decision
  truncation.
- Wall time: 4,611.112 seconds (1h16m51.1s).
- Decisions and information sets: 9,548.
- Winner by simulator net worth: player 1.
- Peak process RSS: 0.52021 GiB.
- Information sets per player: `[2932, 3290, 2832, 494]`.
- Configuration: one simulation per legal action, 256-step rollout horizon,
  epsilon 0.1, 20,000-decision safety cap, seed 0.
- Checkpoint: `artifacts/cfr_ppo_plus/cfr_full_game.pkl.gz` (478.7 KiB).

The completed trajectory was replayed from its stored regret tables and matched
all 9,548 decisions, round count, and winner. A separate average-policy smoke
game loaded the portable checkpoint and completed 200 rounds with player 2 as
winner.

## SHA-256

```text
35b2b58fcaeb2235525650f9420cbff2136d4233acb762778192de123f377d5b  artifacts/ppo_plus/ppo_hybrid_2000.pt
8f955bbed48d28bb8a85955b9e3c547f660c96b9b428cd2b4f1b23d22f632c12  artifacts/ppo_plus/ppo_hybrid_2000_history.json
9fc5eb38dfa1bcc6681cfeeb5fdbf1e680982e48cc8258252e2fdc46d4cc4094  artifacts/cfr_ppo_plus/cfr_full_game.pkl.gz
```
