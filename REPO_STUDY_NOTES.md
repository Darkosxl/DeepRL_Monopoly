# DeepRL Monopoly: Study Notes

These are notes for understanding this repository, not a calendar or a list of chores. The best mental anchor for you is ASU: it is currently the strongest policy you have measured here, and every other agent can be understood as a different answer to the same question:

> Given the current `MonopolyEnv` and its legal actions, which integer action should this player choose?

Your seat-balanced 100-game artifact records 72 ASU wins, 0 Fixed-A wins, 10 Fixed-B wins, and 18 Fixed-C wins. That establishes ASU as the strongest policy in **that experiment**; it is not a universal Monopoly ranking or a reproduction of a paper result.

## 1. The whole repository in one picture

```text
                             ppo-plus-v2 game engine
                         state + rules + legal actions
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
       scripted A-F          PPO / DDQN / CFR           frozen ASU
       hand-written          learned baselines          hand-built value
                                      |                       |
                                      +-----------+-----------+
                                                  |
                                          evaluation arenas

    weak PPO weights + ASU examples + self-play games
                         |
                         v
              MonopolyZero policy/value net
                 + stochastic Max-N PUCT
                         |
                   champion exports
                         |
                         v
              Gemma 4 QLoRA experiment
          state text -> one canonical action JSON
```

The engine is the source of truth. Policies do not directly move pieces or transfer money; they return an action ID, and the engine validates and applies it.

### Repository map

| Area | What it owns | Best entry points |
|---|---|---|
| `monopoly_game_engine/` | Rules, state, legal actions, fixed agents, PPO, DDQN, shared training | `env.py`, `actions.py`, `state.py`, `agents_fixed.py` |
| `ASU_FROZEN_TEACHER/` | Deterministic ASU-inspired value and rollout teachers | `README.md`, `spec.py`, `core.py`, `evaluate.py` |
| `monopoly_bench/` | MonopolyZero search, self-play, replay, gates, releases, teacher exports | `README.md`, `search.py`, `training.py`, `ladder.py` |
| `RL_CFR_MONOPOLYMODIFIED/` | A practical rollout/regret-matching baseline | `.../cfr/classic_cfr.py` |
| `SLM_HANDMADE_MONOPOLY/` | Gemma action schema, ASU dataset creation, QLoRA Colab pipeline | `monopoly_qlora.py`, `Gemma4_12B_Monopoly_QLoRA.ipynb` |
| `tools/` | Training, interactive play, and statistics utilities | `train_and_save.py`, `play_game.py`, `generate_stats.py` |
| `tests/` | Executable contracts and regression checks | Start with tests matching the component you are reading |

## 2. Core vocabulary used everywhere

- **State**: all simulator data needed to continue a game. The neural observation is a 300-float encoding of part of that state.
- **Action**: one integer in `[0, 2957]`. Most integers are illegal in any one state.
- **Legal-action mask**: a Boolean vector that prevents a policy from selecting impossible actions. This is essential, not an optimization.
- **Policy** `pi(a | s)`: a rule or distribution for choosing an action in state `s`.
- **State value** `V(s)`: the expected quality of a state. ASU defines its own explicit value; neural critics learn one.
- **Action value** `Q(s, a)`: expected future return after action `a` in state `s`.
- **Transition**: `env.step(action)` changes the state according to rules and chance.
- **Reward**: a training signal emitted after a transition. It is not the same as final win rate.
- **Episode**: one complete game.
- **On-policy learning**: trains on experience from the policy currently being updated. PPO is on-policy.
- **Off-policy learning**: can train from older behavior stored in replay. DDQN is off-policy.
- **Model-free**: learns without searching a known transition model. PPO and DDQN are model-free.
- **Model-based search**: clones the engine and asks “what happens if...?” ASU rollout and MonopolyZero search do this.
- **Self-play**: agents generate training games against themselves or prior snapshots.
- **Distillation**: a student imitates decisions produced by a teacher. The Gemma pilot is supervised policy distillation, not RL.
- **Four-player stochastic game**: the accurate mathematical setting here. It is more complicated than a single-agent MDP because all four players optimize their own survival and dice create chance nodes.

## 3. The `ppo-plus-v2` game engine

Read [`PPO_PLUS_RULES.md`](PPO_PLUS_RULES.md) as the rules contract and [`monopoly_game_engine/env.py`](monopoly_game_engine/env.py) as the executable version.

### This is a research ruleset, not official tabletop Monopoly

Important deliberate simplifications include:

- no Chance or Community Chest card effects;
- no even-selling constraint or building auctions;
- a property may be sold back to the bank at mortgage value;
- trades are a bounded, enumerable set of offers;
- some debts are limited by the simulator's cash/debt mechanics;
- games stop after 200 rounds and use simulator net worth as the tiebreaker.

`ppo-plus-v2` names the **rules/state/action contract**. It does not mean “version 2 of the PPO algorithm.” A v1 checkpoint is incompatible because v2 changed what the network sees and which output index means which action.

### Turn phases and actors

The normal flow is:

```text
pre_roll -> post_roll -> out_of_turn -> next player's pre_roll
```

Auctions and incoming trades can interrupt that flow. This is why these two methods differ:

- `active_player_id()` is the player whose normal turn is in progress.
- `whose_turn()` is the player who must act now, which may instead be an auction bidder or trade recipient.

Always ask `whose_turn()` before requesting legal actions or a policy decision.

### The 300-dimensional observation

[`monopoly_game_engine/state.py`](monopoly_game_engine/state.py) constructs:

- **240 original features**
  - 4 players x 4 values: position, normalized cash, jail flag, Get Out of Jail card;
  - 28 deeds x 8 values: five ownership slots, mortgage flag, monopoly flag, and development level.
- **60 v2 context features**
  - phase, acting player, active player, dice and doubles;
  - house/hotel bank inventory;
  - bankruptcy, jail, turn order, and debt;
  - auction state and incoming/outgoing trade context;
  - round and extra-roll state.

The player block is **actor-relative**: the acting player appears first, followed by opponents. Most v2 actor/turn indicators use that same order. One legacy wrinkle is that each deed's ownership slots still use physical player IDs 0-3; they do not rotate with the actor. The fifth ownership slot is never set in the current implementation, while an unowned bank deed is represented by all five slots being zero. This mixed convention makes explicit actor-relative/physical-seat conversion important elsewhere, especially for MonopolyZero's four-way win vector.

### Why there are 2,958 actions

| Family | Count |
|---|---:|
| Binary actions such as roll, buy, accept, decline, end turn | 9 |
| Mortgage, development, sale, and related deed operations | 172 |
| Cash-for-property trade offers | 504 |
| Property-for-property exchanges | 2,268 |
| Auction pass and bid increments | 5 |
| **Total** | **2,958** |

This table is the global action **vocabulary**, not a menu of 2,958 simultaneous choices. In an ordinary roll state, `ROLL_DICE` is the only legal action and is therefore forced. It remains an explicit action because `env.step(ROLL_DICE)` triggers the chance transition, and while jailed it can compete with `USE_GOOJ_CARD` or `PAY_BAIL`. The confusingly named pre-roll `END_TURN` action only finishes property/trade management and advances to the roll state; it does not skip the player's turn.

The combinatorial trade encoding causes most of the action space. Uniformly sampling all legal IDs would therefore over-sample trade behavior. The DDQN code compensates during exploration by sampling action **sections**, not just flat IDs.

[`monopoly_game_engine/actions.py`](monopoly_game_engine/actions.py) owns the offsets and the exact ID mapping. Never invent or persist an action ID without this mapping and the ruleset version.

### Legal masks are part of the problem definition

At one moment a player may only be able to roll; at another they may choose among hundreds of trades, mortgages, or building actions. `env.get_allowed_actions(pid)` is authoritative. The neural agents mask illegal logits or Q-values, ASU scores only legal candidates, and evaluators should fail closed if an agent returns an illegal ID.

Forced debt liquidation is especially important: discretionary safety logic must not remove actions needed to pay debt, and bankruptcy only becomes legal when no rescue remains.

### Three notions that look like “value” but are different

1. **Simulator net worth**: cash plus the engine's property/development accounting. It selects a winner at the 200-round cap.
2. **Training potential/reward**: a bounded relative-net-worth signal used for learning and shaping.
3. **ASU value**: assets plus projected rent and monopoly potential; cash is deliberately excluded from its asset term.

Do not compare these numbers as if they shared units or semantics.

## 4. The scripted agents A-F

All live in [`monopoly_game_engine/agents_fixed.py`](monopoly_game_engine/agents_fixed.py). They share deterministic phase-aware machinery but encode different priorities.

| Agent | Personality | Practical behavior |
|---|---|---|
| Fixed-A, The Hoarder | Cash preservation | Large cash buffer, buys mainly monopoly completions/railroads, never develops or trades |
| Fixed-B, The Deal Maker | Aggressive acquisition | Buys with a small buffer, initiates many trades, develops only with abundant cash |
| Fixed-C, The Gambler | High risk | Buys aggressively, makes quick monopoly trades, mortgages as a last resort |
| Fixed-D, The Builder | Development | Prefers strong color groups/railroads and turns monopolies into houses/hotels |
| Fixed-E, The Blocker | Denial | Buys pieces that block opponents and avoids trading threats into monopolies |
| Fixed-F, The Rail Baron | Narrow portfolio | Specializes in railroads/utilities and avoids ordinary color development |

A-C are the traditional training/evaluation trio. D-F widen the strategic coverage used by the MonopolyZero gate. They are useful baselines because their behavior is interpretable, stable, cheap, and free of checkpoint compatibility issues.

## 5. ASU: the hand-built teacher

Start with [`ASU_FROZEN_TEACHER/README.md`](ASU_FROZEN_TEACHER/README.md), then [`spec.py`](ASU_FROZEN_TEACHER/spec.py), then [`core.py`](ASU_FROZEN_TEACHER/core.py).

Both policies are explicitly **ASU-inspired reconstructions for this repository**, not the original paper agent and not a claim to reproduce its win rates.

### `asu_value_v1`

For player `i`, the core heuristic is:

```text
V_i(s) = M_assets + R_short + R_long + M_monopoly
```

| Term | Meaning |
|---|---|
| `M_assets` | List price plus development cost of the player's unmortgaged holdings. Cash is excluded. |
| `R_short` | Expected rent received minus expected rent paid across every player's next five complete turns. |
| `R_long` | A coarse five-lap rent projection: `40/7` landing opportunities per lap, uniform over the 28 deeds. |
| `M_monopoly` | Future value of completing and developing color groups with projected funds, bank inventory, and the v2 building rules. |

The short projection is much less crude than “average dice = 7.” It enumerates ordered dice outcomes and models doubles, triple doubles, jail, Go To Jail, railroad ownership counts, and utility dice totals.

Monopoly potential assumes available funds of:

```text
current cash + 5 * $200 Go salary + R_long
```

It prices missing deeds at list price, searches feasible development under the 32-house/12-hotel bank inventory, and discounts the result by `2 ** missing_deeds`. A nearly complete group is therefore worth much more than a speculative group missing several properties.

### How one ASU decision is made

For every legal action, ASU clones the environment, applies the action, evaluates the resulting state, and records an immutable candidate breakdown. Roll actions average all 36 ordered two-die outcomes. Exact ties use a semantic priority and then the lower action ID, so repeated runs agree.

Discretionary spending must pass both bankruptcy gates:

```text
cash_after + next_round_net_rent >= $200

cash_after + next_round_rent_income
           + liquidatable_worth
           - worst_reachable_rent > 0
```

Forced actions and debt-liquidation actions are never pruned. If every discretionary choice is unsafe, ASU takes the safest legal progress action.

Special cases are evaluated semantically:

- **Trade**: simulate the accepted transfer; proposer gain must be `> 0`, recipient gain must be `>= 0`, and safety must pass.
- **Auction**: derive a ceiling from marginal ASU value and safety, then use the largest legal increment below it; otherwise pass.

`decide(env)` is valuable for debugging and labels because it returns every candidate's score, four value components, safety rejection, and the frozen-spec hash. `choose_action(env)` returns only the selected integer.

### `asu_rollout_v1`

This is strength-first truncated lookahead:

1. Rank legal actions with `asu_value_v1`.
2. Keep at most eight, while preserving mandatory safety actions.
3. Run eight simulations per candidate.
4. Let every seat use `asu_value_v1` for the next 32 decisions.
5. Use terminal utility if the game ends; otherwise score the leaf with the root player's ASU value.

All candidates use the same seed streams. This is **common random numbers**: action A and action B face comparable dice luck, reducing comparison noise. The code also restores Python/NumPy global RNG state and never mutates the caller's environment.

ASU is strong here because it starts with substantial Monopoly knowledge: rent exposure, development, cash safety, trades, and auctions. PPO/DDQN must discover those delayed relationships from games. ASU's weakness is the mirror image: its assumptions are frozen and hand-designed, so it cannot naturally learn a strategy outside them.

## 6. DDQN: learned action values

Read [`monopoly_game_engine/agent_ddqn.py`](monopoly_game_engine/agent_ddqn.py) with [`networks.py`](monopoly_game_engine/networks.py).

The network maps 300 state features through hidden layers `1024 -> 512` to 2,958 Q-values. With legal next actions `L(s')`, the Double-DQN target is conceptually:

```text
a* = argmax over a in L(s') of Q_online(s', a)
target = r + gamma * Q_target(s', a*)
```

The online network **selects** the next action and the delayed target network **evaluates** it. This separation reduces the maximization overestimation seen in ordinary DQN.

Key mechanisms:

- **Replay buffer** breaks temporal correlation and reuses old transitions.
- **Epsilon-greedy exploration** sometimes chooses a random legal action.
- **Section-balanced exploration** avoids drowning in the 2,268 exchange IDs.
- **Target network** changes slowly, stabilizing the bootstrapped target.
- **Hybrid mode** may intercept selected buy/trade decisions with scripted logic; intercepted actions are not proof that the neural policy learned them.

Your original completed 2,000-game v2 DDQN artifact records 0/200 wins against Fixed-A/B/C and frequent trade/mortgage cycling. The useful lesson is not “DDQN can never play Monopoly.” It is that sparse delayed credit, a huge structured action space, opponent non-stationarity, and exploitable shaping signals make this particular training setup hard.

## 7. PPO: learned action probabilities plus a critic

Read [`monopoly_game_engine/agent_ppo.py`](monopoly_game_engine/agent_ppo.py), [`networks.py`](monopoly_game_engine/networks.py), and [`train.py`](monopoly_game_engine/train.py).

PPO has two learned functions:

- the **actor** outputs a masked action distribution `pi(a | s)`;
- the **critic** estimates expected return `V(s)`.

The probability ratio for a stored action is:

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

PPO maximizes the smaller of the normal advantage objective and a clipped version of that ratio. Clipping discourages one batch from moving the policy too far from the behavior that generated it.

**GAE**, generalized advantage estimation, combines multi-step temporal-difference errors. Its parameter trades some bias for lower variance. In this repo, a mid-game update bootstraps from the critic because the episode has not actually terminated.

The shared trainer lets one learner act against Fixed-A/B/C. A neural transition spans intervening opponent/scripted decisions until the learner acts again. Reward shaping uses the change in relative-net-worth potential plus terminal win/loss reward.

The completed v2 PPO run documented in [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) is a valid but weak baseline. An older chart showing a high PPO win rate belongs to a different rules/state/action setup and is not evidence for v2 performance. MonopolyZero may copy PPO actor weights as an initialization; that does not make PPO a runtime dependency or a strong teacher.

## 8. CFR-style regret matching

The current implementation is [`RL_CFR_MONOPOLYMODIFIED/RL_models_1_CounterfactualRegretMinimization/cfr/classic_cfr.py`](RL_CFR_MONOPOLYMODIFIED/RL_models_1_CounterfactualRegretMinimization/cfr/classic_cfr.py).

An **information set** groups situations the strategy treats as equivalent. Each information set stores:

- cumulative regret for each legal action;
- cumulative strategy probabilities.

**Regret matching** assigns probability proportional to positive cumulative regret. If no action has positive regret, it uses a uniform distribution. The reported policy is the time-averaged strategy.

This repo's trainer samples rollouts for legal actions and updates regret by:

```text
regret(action) += utility(action) - expected_utility(current_strategy)
```

Important caveat: it omits formal counterfactual reach weighting and MCCFR importance corrections. It is therefore a practical CFR-style rollout baseline, **not** a proven equilibrium solver. Its large information table and per-action rollouts also make it expensive.

## 9. MonopolyZero: policy/value learning plus search

Read [`monopoly_bench/README.md`](monopoly_bench/README.md), then [`model.py`](monopoly_bench/model.py), [`search.py`](monopoly_bench/search.py), and [`training.py`](monopoly_bench/training.py).

### Network

`MonopolyZeroNet` begins from a PPO-shaped actor trunk and policy head, then adds a four-way win-probability head. It predicts:

- a prior probability over legal actions;
- one win probability for each physical player.

The four-value output matters because Monopoly is not two-player zero-sum. A single scalar “good for current player / bad for opponent” cannot represent three distinct opponents.

### Stochastic Max-N PUCT

At a decision node, each edge stores visits, prior, and a four-player value vector. The player acting at that node selects using its own component:

```text
score(a) = Q_current_actor(a)
         + c_puct * prior(a) * sqrt(parent_visits) / (1 + edge_visits)
```

This is **Max-N**: when the actor changes, the component being maximized changes. It is not two-player minimax.

Rolls become explicit chance nodes with all 36 ordered dice outcomes. Search clones the engine, so looking ahead must not advance the real game or contaminate global RNG state.

### Progressive widening

Expanding all 2,958 actions at every node would waste search. The tree initially exposes a small subset, preserves key binary actions and the best prior from each action family, then widens as visit count grows. This balances strategic coverage with compute.

### What the network learns from

- **Policy target**: normalized MCTS visit counts, not merely the one chosen action.
- **Value target**: the actual game's one-hot winner, expressed relative to the actor and converted correctly to physical seats.
- **ASU bootstrap target**: cross-entropy imitation of ASU's action; the value target still comes from the real winner, not ASU's heuristic score.

The lifecycle is:

```text
weak PPO warm start
  -> ASU imitation/bootstrap games
  -> self-play and games versus snapshots/baselines
  -> replay training
  -> candidate versus incumbent arena
  -> promote only if statistical and safety gates pass
```

ASU imitation decays to zero over eight generations. A released MonopolyZero model has no runtime ASU dependency. ASU teaches useful early behavior; self-play and search are intended to surpass or correct it.

Replay uses disk-backed/memory-mapped storage, sparse MCTS targets, packed legal masks, atomic checkpoints, and deterministic reconciliation. These are engineering answers to large data, interruption, and reproducibility—not new RL algorithms.

## 10. Gemma 4 QLoRA: compressing decisions into text generation

[`SLM_HANDMADE_MONOPOLY/monopoly_qlora.py`](SLM_HANDMADE_MONOPOLY/monopoly_qlora.py) defines the contract. [`Gemma4_12B_Monopoly_QLoRA.ipynb`](SLM_HANDMADE_MONOPOLY/Gemma4_12B_Monopoly_QLoRA.ipynb) runs the heavy Colab experiment.

The model receives a compact actor-relative JSON state plus grouped legal domains. It must emit exactly one canonical JSON object such as a roll, mortgage, trade, or auction bid. The parser requires exact keys, converts the object back to an action ID, and checks legality. Invalid text triggers a deterministic safe fallback and is counted as a failure.

### Dataset logic

- ASU scores legal candidates and supplies the supervised label.
- Occasionally, a safe alternative is executed to visit different states, but the recorded label remains ASU's choice.
- Duplicate state prompts are removed.
- Entire games, not individual rows, are assigned to train/validation/test to prevent trajectory leakage.
- Rows are balanced across phase and action family.
- Only completion tokens contribute to loss; prompt labels are `-100`.
- Truncation is forbidden if more than 0.5% of rows exceed the 512-token context.
- Code, ruleset, configuration, state, and teacher bundles are hashed.

### LoRA and QLoRA

**LoRA** freezes a pretrained weight matrix and learns a low-rank update, roughly `Delta W = B A`. Rank `r` controls adapter capacity; `alpha` scales the update.

**QLoRA** keeps the frozen base model quantized to 4-bit while gradients update the small LoRA matrices. This cuts memory dramatically compared with full-parameter fine-tuning.

The pilot freezes the Gemma 4 12B base, trains attention-only LoRA adapters with rank/alpha 4, uses sequence length 512, micro-batch 1 with eight-step gradient accumulation, and runs one epoch. Unsloth supplies memory-oriented model loading/checkpointing; TRL's `SFTTrainer` runs supervised fine-tuning.

Before game evaluation, the adapter must meet offline gates: at least 98% parseable JSON, 97% legal output without fallback, and 65% exact ASU agreement. Exact agreement is an imitation metric; game win rate is the real policy metric.

## 11. Evaluation and experimental discipline

For four-player Monopoly, a raw win count is meaningful only with its setup.

- **Seat balance**: rotate the focus policy through every physical seat.
- **Paired seeds**: compare policies under matched game randomness.
- **Turn order**: distinguish randomized turn order from physical seat assignment.
- **Opponent identity**: “72%” is incomplete without saying “against Fixed-A/B/C under ppo-plus-v2.”
- **Round cap**: report truncations and the net-worth tiebreaker.
- **Uncertainty**: a Wilson interval communicates how noisy a finite win rate is.
- **Checkpoint identity**: hash files and reject state/action/ruleset mismatch.
- **Source identity**: ASU and MonopolyZero freeze specification/source hashes so future results can identify exactly what ran.
- **Determinism**: clone environments and preserve global RNG state during hypothetical rollouts.
- **Fail closed**: an illegal model action is a policy failure, not permission for the evaluator to silently pick a strong replacement.

The two result artifacts most relevant to your recent comparisons are:

- [`artifacts/asu_frozen_teacher/asu_value_v1_vs_fixed_abc_100/summary.json`](artifacts/asu_frozen_teacher/asu_value_v1_vs_fixed_abc_100/summary.json)
- [`artifacts/ddqn_plus/ddqn_hybrid_2000_v2_eval_stats.json`](artifacts/ddqn_plus/ddqn_hybrid_2000_v2_eval_stats.json)

Heavy training, broad search, and Gemma QLoRA belong in Colab for this project. The `*_output.ipynb` files are captured executions, not core source. Local inspection, unit tests, and small smoke checks are different from launching a full training/evaluation workload.

## 12. Trace one decision through the code

This is the most useful code-reading trace:

1. `pid = env.whose_turn()` determines the real actor.
2. `legal = env.get_allowed_actions(pid)` derives valid IDs from phase, debt, trade, auction, cash, and property state.
3. A policy encodes or reads the state:
   - fixed agent: ordered semantic rules;
   - ASU: clone and score every legal result;
   - DDQN: masked Q argmax;
   - PPO: masked actor distribution;
   - CFR: sample average regret-matched strategy;
   - MonopolyZero: MCTS guided by network priors/values;
   - Gemma: serialize prompt, generate JSON, parse to ID.
4. `env.step(action)` rejects illegality and applies the rules.
5. The engine may move to another phase, another physical player, an auction bidder, or a trade recipient.
6. During learning, the transition becomes PPO trajectory data, a DDQN replay tuple, a MonopolyZero search position, or an ASU/Gemma teacher row.

If you understand why each policy produces a different answer at step 3 while sharing steps 1, 2, and 4, you understand the architecture.

## 13. Common traps

- `ppo-plus-v2` is a ruleset/schema version, not an improved PPO model.
- A graph from v1 cannot establish v2 checkpoint strength.
- ASU value is not simulator net worth.
- `asu_rollout_v1` is truncated policy rollout, not MCTS.
- MonopolyZero is AlphaZero-inspired, but adapted for four selfish players, dice, and a huge legal action set.
- The CFR baseline is CFR-style; it lacks the machinery needed for an equilibrium guarantee.
- A PPO warm start does not make MonopolyZero depend on the old PPO folder at inference.
- ASU's 72/100 result is strong evidence for that exact matchup, not a paper reproduction.
- Exact teacher agreement is not the same as winning games.
- Random seeds help reproducibility only when every RNG stream and clone boundary is controlled.

## 14. Compact self-test

You have the repo's core ideas if you can answer these without looking:

1. Why can `whose_turn()` differ from `active_player_id()`?
2. Why does the trade encoding dominate the 2,958-action space?
3. Why must every neural output be legal-masked?
4. Why does ASU exclude cash from `M_assets` but still use cash in safety and monopoly planning?
5. What does Double DQN separate between the online and target networks?
6. Why is PPO on-policy while DDQN can reuse replay?
7. Why is the current CFR implementation not a guaranteed equilibrium solver?
8. Why does MonopolyZero predict four values instead of one?
9. Why are roll actions chance nodes with 36 ordered outcomes?
10. What does progressive widening solve?
11. Why must dataset splits keep whole games together?
12. Why are JSON legality rate and game win rate both needed for Gemma?

## 15. Further reading

### Papers closest to this repository

- [Open-world Monopoly solver / ASU foundation](https://arxiv.org/abs/2107.04303)
- [Interactive novelty handling / follow-up architecture](https://arxiv.org/abs/2302.14208)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [AlphaZero](https://arxiv.org/abs/1712.01815)
- [Counterfactual Regret Minimization](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)
- [Monte Carlo CFR](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)

### Current library documentation

- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [inference mode](https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html), and [saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [NumPy random generation](https://numpy.org/doc/stable/reference/random/index.html) and [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [Transformers chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) and [generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [TRL `SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Unsloth Gemma 4 training](https://unsloth.ai/docs/models/gemma-4/train)
