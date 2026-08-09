# PPO-plus Monopoly rules

`ppo-plus-v1` is the one canonical game used by the new PPO and CFR paths. PPO
uses the engine directly; CFR clones and explores that same engine through
`classic_cfr.py`. Algorithm-specific policy and checkpoint code remains
separate.

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
- Property-for-cash and property-for-property offers are supported. The PPO
  fixed opponents use their individual buying personalities during auctions.
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
  60 values add phase, actor, dice/doubles, inventory, debt, auction, and round
  context.
- Action space: 2,958 actions. The original 2,953 IDs are preserved and five
  auction actions are appended.
- Checkpoints record the ruleset, state dimension, and action dimension. Old
  PPO and CFR checkpoints fail with an explicit incompatibility error instead
  of loading with the wrong network or table shape.

## Training and play

From the repository root, train the hybrid PPO model for 2,000 games:

```bash
python 'RL_PPO(UNOFFICIAL)_MONOPOLY/train_and_save.py' \
  --algo ppo --games 2000 --device auto \
  --out artifacts/ppo_plus/ppo_hybrid_model.pt
```

The trainer checkpoints every 100 games. It saves and stops at 3 GiB process
RSS, refuses to continue at 4 GiB, and stops when system-available RAM reaches
2 GiB. Thresholds are configurable with the three memory CLI flags. The JSON
history records elapsed time and peak CPU/GPU memory.

Play a trained PPO checkpoint:

```bash
python 'RL_PPO(UNOFFICIAL)_MONOPOLY/play_game.py' \
  --algo ppo --players 4 \
  --model artifacts/ppo_plus/ppo_hybrid_model.pt
```

Train one four-player Monte Carlo CFR game:

```bash
python -m \
  RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr \
  train --games 1 \
  --checkpoint artifacts/cfr_ppo_plus/cfr.pkl.gz
```

CFR keeps four separate regret/average-strategy tables and evaluates every
currently legal action at each visited decision. It samples chance and uses
finite rollouts because the full Monopoly game tree is intractable; therefore
this is Monte Carlo CFR, not an exact full-tree equilibrium computation. CFR
reports progress every 10 decisions and atomically saves partial regret tables
every 100 decisions by default. An interrupted run also saves its tables. CFR
checkpoints use Python pickle and should only be loaded from trusted sources.

Play games from the learned average CFR policy:

```bash
python -m \
  RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr \
  play artifacts/cfr_ppo_plus/cfr.pkl.gz --games 10
```
