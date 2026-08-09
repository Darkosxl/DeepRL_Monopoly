# Shared Monopoly rules

`ppo-plus-v2` is the historical compatibility identifier for the one canonical
game used by the DDQN, PPO, and CFR paths. DDQN and PPO use the engine directly;
CFR clones and explores that same engine through `classic_cfr.py`.
Algorithm-specific policy and checkpoint code remains separate.

This is a classic-board research ruleset, not an exact implementation of every
official Monopoly rule. The name is deliberately explicit so results are not
presented as traditional-rule Monopoly.

## Rules implemented

- Four players, the standard 40-space US board, standard deed prices and rents,
  $1,500 starting cash, $200 for passing Go, taxes, jail, Free Parking, and Go
  To Jail.
- Property ownership, railroad and utility scaling, doubled unimproved rent for
  a complete color group, mortgages, unmortgaging, houses, hotels, and rent.
- Doubles grant another roll, three consecutive doubles send the player to
  jail, and doubles rolled in jail release the player without another roll.
- Every declined or unaffordable unowned deed enters a cash auction. Bids use
  fixed +$1, +$10, +$50, and +$100 actions; passing withdraws a bidder.
- A finite bank holds 32 houses and 12 hotels. Building, selling, and
  bankruptcy return the corresponding pieces.
- Unpaid rent creates an explicit player creditor. The debtor may liquidate;
  bankruptcy transfers remaining cash, deeds, mortgage state, and a jail card
  to that creditor. Bank debt returns deeds to the bank.
- Property-for-cash and property-for-property offers are supported. The fixed
  opponents use their individual buying personalities during auctions.
- The engine rejects actions outside the current legal-action mask, including
  premature turn endings and bankruptcy declarations by solvent players.
- Games end when one player remains or after 200 rounds; a capped game is won
  by the greatest simulator net worth.

## Deliberate differences from traditional Monopoly

- Chance and Community Chest spaces have no card effect. There is no shuffled
  card deck, so Get Out of Jail Free cards are not normally introduced.
- Houses and hotels need a complete color group, but even building and even
  selling across that group are not enforced. Building auctions are omitted.
- The simulator permits selling an undeveloped deed back to the bank at its
  mortgage value. Traditional Monopoly normally uses mortgages or player
  trades instead.
- Mortgage and building checks are per deed rather than enforcing every
  color-group restriction from the official rules.
- Trade actions use a bounded research action space: cash offers are 75%, 100%,
  or 125% of list price, or one deed is exchanged for one deed.
- Income and luxury tax payments are limited to cash on hand and do not create
  a liquidation phase. Jail's forced third-turn payment is also limited to cash
  on hand.
- The 200-round cap and simulator net-worth tie-break are research controls,
  not traditional rules.

## Public dimensions and compatibility

- Observation: 300 float values. The original 240-value prefix is preserved;
  60 values add phase, actor, dice/doubles, inventory, bankruptcy, jail-turn,
  turn-order, debt, auction, round, and actionable trade context.
- Action space: 2,958 actions. The original 2,953 IDs are preserved and five
  auction actions are appended.
- Checkpoints record the ruleset, state dimension, and action dimension. Old
  DDQN, PPO, and CFR checkpoints fail with an explicit incompatibility error
  instead of loading with the wrong network or table shape.

## Training and play

From the repository root, train a hybrid DDQN milestone for 2,000 games:

```bash
python tools/train_and_save.py \
  --algo ddqn --games 2000 --device auto \
  --out artifacts/ddqn_plus/ddqn_hybrid_2000_v2.pt
```

The DDQN follows the paper's 1,024/512 ReLU network, 0.9999 discount, 1e-5
learning rate, 128-sample batches, 10,000-transition replay, and 500-game target
updates. The paper trained for 10,000 games; 2,000 is an intermediate checkpoint,
not a reproduction of its final result.

The shared trainer uses potential-difference shaping between neural decisions
plus a terminal win/loss reward. It checkpoints every 100 games, saves and stops
at 3 GiB process RSS, refuses to continue at 4 GiB, and stops when
system-available RAM reaches 2 GiB. Thresholds are configurable with the three
memory CLI flags. The JSON history records elapsed time and peak CPU/GPU memory.

Resume a compatible DDQN checkpoint to a total game count with the same output
path and seed:

```bash
python tools/train_and_save.py \
  --algo ddqn --games 10000 --device auto --seed 42 --resume \
  --out artifacts/ddqn_plus/ddqn_hybrid_2000_v2.pt
```

Format-three checkpoints include optimizer state, training configuration,
replay memory, learned steps, and completed games. Game-indexed seeding makes a
game-boundary resume reproduce an uninterrupted run.

Play a trained DDQN checkpoint:

```bash
python tools/play_game.py \
  --algo ddqn --players 4 \
  --model artifacts/ddqn_plus/ddqn_hybrid_2000_v2.pt
```

PPO remains available through `--algo ppo` for baseline comparison and old
compatible workflows; it is no longer the recommended training path.

Train one four-player CFR-style rollout regret game:

```bash
python -m \
  RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr \
  train --games 1 \
  --checkpoint artifacts/cfr_ppo_plus/cfr.pkl.gz
```

The trainer keeps four separate regret/average-strategy tables and evaluates
every currently legal action at each visited decision using finite sampled
rollouts. It performs direct rollout regret matching without counterfactual
reach weighting or MCCFR sampling corrections. It is therefore a practical
CFR-style approximation, not formal MCCFR, and carries no equilibrium
guarantee. It reports progress every 10 decisions and atomically saves every
100 decisions by default. Format-three checkpoints preserve the exact active
game, RNG, decision count, and elapsed time, so `--resume` continues an
interrupted trajectory. CFR checkpoints use Python pickle and should only be
loaded from trusted sources.

Play games from the learned average CFR policy:

```bash
python -m \
  RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr \
  play artifacts/cfr_ppo_plus/cfr.pkl.gz --games 10
```
