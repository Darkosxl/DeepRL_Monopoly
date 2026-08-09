# PPO-plus training results

Measured on 2026-08-09 with an NVIDIA GeForce RTX 4050 Laptop GPU. Generated
checkpoints are under `artifacts/`, which is intentionally ignored by Git.

## Hybrid PPO v2: 2,000 games

- Result: all 2,000 games completed.
- Wall time: 1,880.325 seconds (31m20.3s).
- Mean wall time: 0.94016 seconds/game.
- Peak process RSS: 1.63732 GiB.
- Peak CUDA allocation: 0.04536 GiB.
- Final 40-game win-rate window: 0.0%.
- Best 40-game win-rate window: 2.5%.
- Learned policy steps: 1,407,156.
- Checkpoint: `artifacts/ppo_plus/ppo_hybrid_2000_v2.pt`
  (14,205,587 bytes; 13.5 MiB).
- History: `artifacts/ppo_plus/ppo_hybrid_2000_v2_history.json`
  (4,838 bytes; 4.7 KiB).

The format-three checkpoint reports `ppo-plus-v2`, 300 observations, 2,958
actions, and 2,000 completed games. It loaded successfully on CPU. A seeded
four-player inference smoke game also completed; the learned player went
bankrupt and player 4 won. The model is a valid training and inference
baseline, but its measured win rate is weak and it should not be presented as
a strong policy. One smoke game is not a statistically useful evaluation.

## Four-player CFR-style rollout regret matching v2: one full game

- Result: completed at the configured 200-round cap without decision
  truncation.
- Wall time: 1,532.770 seconds (25m32.8s).
- Decisions and information sets: 12,833.
- Winner by simulator net worth: player 1.
- Peak process RSS: 0.52294 GiB.
- Information sets per player: `[3314, 3160, 3224, 3135]`.
- Configuration: one simulation per legal action, 256-step rollout horizon,
  epsilon 0.1, 20,000-decision safety cap, seed 0.
- Checkpoint: `artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz`
  (495,865 bytes; 484.2 KiB).

The format-three checkpoint loaded with one completed game, no in-progress
trajectory, and 12,833 portable information sets. A separate average-policy
smoke game completed 200 rounds and 9,016 decisions without truncation, with
player 1 as winner. This trainer uses direct finite-rollout regret matching; it
is not formal MCCFR and has no equilibrium guarantee. One trajectory is a
functional baseline, not evidence of convergence or strong play.

The older unsuffixed PPO and CFR artifacts were produced under
`ppo-plus-v1`. They are retained for comparison, but the v2 loaders reject them
as incompatible and their earlier performance figures are stale.

## SHA-256

```text
4c364204eb59df74dffab911f8fbde523e59037558fafbe49daaf79e5c9180db  artifacts/ppo_plus/ppo_hybrid_2000_v2.pt
71f6f69ac2efe3355fc003e17ea6df9ba0d112a50a7744bde3f2c8196a9215f6  artifacts/ppo_plus/ppo_hybrid_2000_v2_history.json
3cbd5c88e8fd5e0d9c2666261b81488c224d2498d16142c652ba7a43806e80f4  artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz
```
