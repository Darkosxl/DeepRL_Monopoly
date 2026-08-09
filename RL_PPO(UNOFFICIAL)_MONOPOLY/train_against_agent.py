"""
train_against_agent.py
----------------------
Train a DRL agent against a configurable mix of DRL opponents and
fixed-policy opponents.

Default
-------
  1 DRL opponent (fresh, co-trains alongside the main agent)
  2 fixed-policy opponents (FPAgentA, FPAgentB, FPAgentC in rotation)

DRL opponent modes
------------------
  new          Fresh agent that learns at the same time as the main agent
               (co-training / self-play).
  pretrained   Load weights from a .pt file; the opponent is frozen and does
               not learn.

Usage examples
--------------
  # Default: 1 co-training DRL opponent + 2 fixed opponents
  python train_against_agent.py

  # 2 co-training DRL opponents + 1 fixed opponent
  python train_against_agent.py --n-drl 2

  # 3 co-training DRL opponents, no fixed opponents
  python train_against_agent.py --n-drl 3

  # 1 frozen pre-trained DRL opponent + 2 fixed opponents
  python train_against_agent.py --n-drl 1 --drl-mode pretrained --rival model.pt

  # 2 frozen pre-trained opponents + 1 fixed
  python train_against_agent.py --n-drl 2 --drl-mode pretrained --rival model.pt

  # All fixed opponents (same as original train.py)
  python train_against_agent.py --n-drl 0

  # All co-training DRL opponents
  python train_against_agent.py --n-drl 3

  # DDQN main agent, more training games
  python train_against_agent.py --algo ddqn --games 10000

  # Custom output file
  python train_against_agent.py --n-drl 2 --out my_agent.pt
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from monopoly_drl import DDQNAgent, PPOAgent
from monopoly_drl.actions import OFFSETS, ActionType
from monopoly_drl.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_drl.constants import NUM_PLAYERS
from monopoly_drl.env import MonopolyEnv

# ── Action classification (mirrors train.py) ──────────────────────────────────


def _classify_action(a: int):
    """Return (is_buy, is_accept, is_decline, is_trade_offer) flags."""
    buy_off = int(ActionType.BUY_PROPERTY)
    acc_off = int(ActionType.ACCEPT_TRADE)
    dec_off = int(ActionType.DECLINE_TRADE)
    is_offer = (
        OFFSETS["buy_trade"] <= a < OFFSETS["buy_trade"] + 252
        or OFFSETS["sell_trade"] <= a < OFFSETS["sell_trade"] + 252
        or OFFSETS["exch_trade"] <= a < OFFSETS["exch_trade"] + 2268
    )
    return (a == buy_off, a == acc_off, a == dec_off, is_offer)


# ── Agent factories ────────────────────────────────────────────────────────────


def _make_agent(algo: str, hybrid: bool, player_id: int):
    """Create a fresh learning agent."""
    if algo == "ppo":
        return PPOAgent(player_id=player_id, hybrid=hybrid)
    return DDQNAgent(player_id=player_id, hybrid=hybrid)


def _load_agent(algo: str, hybrid: bool, player_id: int, path: str):
    """Load a pre-trained agent from a .pt checkpoint."""
    agent = _make_agent(algo, hybrid, player_id)
    agent.load(path)
    print(f"  Loaded rival weights from {path!r} → player {player_id}")
    return agent


# ── Episode runner ─────────────────────────────────────────────────────────────


def run_episode(
    env: MonopolyEnv,
    main_pid: int,
    main_agent,
    drl_opponents: Dict[int, object],  # pid → learning agent (may be empty)
    fixed_opponents: Dict[int, object],  # pid → fixed-policy agent (may be empty)
    update_drl_opponents: bool,
    is_ppo: bool,
    update_online: bool = True,
) -> Dict:
    """
    Run one complete game.

    Seat layout:
      - main_pid          → main_agent (always learning)
      - drl_opponents     → DRL agents; trained if update_drl_opponents=True
      - fixed_opponents   → fixed-policy agents (never learn)

    Returns a metrics dict compatible with train.py's run_episode().
    """

    # Per-seat state snapshots (each agent sees its own perspective)
    all_pids = list(range(NUM_PLAYERS))
    states = {p: env._get_state(p) for p in all_pids}
    prev_states = dict(states)
    prev_actions = {p: None for p in all_pids}
    prev_log_probs = {p: 0.0 for p in all_pids}
    prev_values = {p: 0.0 for p in all_pids}

    done = False
    total_reward = 0.0
    steps = 0
    update_stats = {}

    # Main-agent metric counters
    trades_initiated = 0
    trades_accepted = 0
    trades_declined = 0
    properties_acquired = 0
    prev_prop_count = len(env.players[main_pid].properties)

    max_steps = env.max_rounds * NUM_PLAYERS * 30
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        pid = env.whose_turn()

        if env.players[pid].bankrupt:
            env._advance_turn()
            continue

        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.DO_NOTHING)]

        state = env._get_state(pid)

        # ── Choose action ──────────────────────────────────────────────────────
        if pid == main_pid:
            if is_ppo:
                action, log_prob, value = main_agent.choose_action(state, env, allowed)
            else:
                action = main_agent.choose_action(state, env, allowed)
                log_prob, value = 0.0, 0.0

        elif pid in drl_opponents:
            agent = drl_opponents[pid]
            if is_ppo:
                action, log_prob, value = agent.choose_action(state, env, allowed)
            else:
                action = agent.choose_action(state, env, allowed)
                log_prob, value = 0.0, 0.0

        else:
            # Fixed-policy opponent
            fp_agent = fixed_opponents[pid]
            action = fp_agent.choose_action(env)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else allowed[0]
                )
            log_prob, value = 0.0, 0.0

        # ── Track main-agent metrics ───────────────────────────────────────────
        if pid == main_pid:
            is_buy, is_acc, is_dec, is_offer = _classify_action(action)
            if is_buy:
                properties_acquired += 1
            elif is_acc:
                trades_accepted += 1
            elif is_dec:
                trades_declined += 1
            elif is_offer:
                trades_initiated += 1

        # ── Step the environment ───────────────────────────────────────────────
        _, _, done, _ = env.step(action)

        # Refresh per-agent states and rewards
        for p in all_pids:
            states[p] = env._get_state(p)
        reward_for = {p: env._compute_reward(p) for p in all_pids}

        # Detect trade-based property gain for main agent
        if pid == main_pid:
            new_count = len(env.players[main_pid].properties)
            if action == int(ActionType.ACCEPT_TRADE) and new_count > prev_prop_count:
                properties_acquired += new_count - prev_prop_count
            prev_prop_count = new_count
            total_reward += reward_for[main_pid]
            steps += 1

        # ── Store / update learning agents ────────────────────────────────────
        if update_online:
            # Determine which learning agents to update this step
            learning_seats = {main_pid: main_agent}
            if update_drl_opponents:
                learning_seats.update(drl_opponents)

            for p, agent in learning_seats.items():
                r_p = reward_for[p]
                ps = prev_states[p]
                ns = states[p]
                pa = prev_actions[p]

                if is_ppo:
                    if p == pid:  # only the agent that just acted gets stored
                        lp = log_prob if p == pid else prev_log_probs[p]
                        v = value if p == pid else prev_values[p]
                        agent.store(ps, action, lp, r_p, v, done)
                        if len(agent.buffer) >= agent.n_steps:
                            stats = agent.update()
                            if p == main_pid:
                                update_stats = stats
                else:
                    # DDQN: store (prev_s, prev_a, r, next_s, done) tuples
                    if p == pid and pa is not None:
                        agent.store_transition(ps, pa, r_p, ns, done)
                    stats = agent.update()
                    if p == main_pid:
                        update_stats = stats

        # ── Bookkeeping ────────────────────────────────────────────────────────
        prev_log_probs[pid] = log_prob
        prev_values[pid] = value
        prev_states[pid] = states[pid]
        prev_actions[pid] = action

    # ── Game over ──────────────────────────────────────────────────────────────
    winner = env.winner()

    if update_online:
        learning_seats = {main_pid: main_agent}
        if update_drl_opponents:
            learning_seats.update(drl_opponents)

        for p, agent in learning_seats.items():
            won_p = winner == p
            agent.add_win_loss(won_p)

            if is_ppo:
                if len(agent.buffer) > 0:
                    stats = agent.update()
                    if p == main_pid:
                        update_stats.update(stats)
            else:
                if prev_actions[p] is not None:
                    agent.store_transition(
                        prev_states[p],
                        prev_actions[p],
                        agent.win_loss_bonus * (1 if won_p else -1),
                        states[p],
                        True,
                    )

    return {
        "won": (winner == main_pid),
        "reward": total_reward,
        "steps": steps,
        "stats": update_stats,
        "trades_initiated": trades_initiated,
        "trades_accepted": trades_accepted,
        "trades_declined": trades_declined,
        "properties_acquired": properties_acquired,
    }


# ── Main training function ─────────────────────────────────────────────────────


def train_mixed(
    algo: str,
    hybrid: bool,
    n_drl: int = 1,
    drl_mode: str = "new",  # "new" | "pretrained"
    rival_path: Optional[str] = None,
    n_games: int = 2000,
    log_every: int = 50,
    seed: int = 42,
) -> tuple:
    """
    Train the main agent (player 0) against a mix of DRL and fixed opponents.

    Parameters
    ----------
    algo        : "ppo" or "ddqn"
    hybrid      : Whether to use hybrid fixed-rule mode
    n_drl       : Number of DRL opponents (0–3). The remaining seats are filled
                  with fixed-policy agents cycling through FPAgentA/B/C.
    drl_mode    : "new"        — DRL opponents are freshly initialised and
                                 co-train alongside the main agent.
                  "pretrained" — DRL opponents load weights from rival_path
                                 and are frozen (do not learn).
    rival_path  : Path to a .pt checkpoint; required when drl_mode="pretrained".
    n_games     : Total training games.
    log_every   : Games between progress log lines.
    seed        : Random seed.

    Returns
    -------
    (main_agent, history_dict)
    """
    if n_drl < 0 or n_drl > NUM_PLAYERS - 1:
        raise ValueError(f"n_drl must be between 0 and {NUM_PLAYERS - 1}, got {n_drl}")
    if drl_mode == "pretrained" and not rival_path:
        raise ValueError("rival_path is required when drl_mode='pretrained'")
    if rival_path and not os.path.exists(rival_path):
        raise FileNotFoundError(f"Rival weights not found: {rival_path!r}")

    random.seed(seed)
    np.random.seed(seed)

    is_ppo = algo == "ppo"
    main_pid = 0
    other_pids = [p for p in range(NUM_PLAYERS) if p != main_pid]  # [1, 2, 3]

    # ── Assign seats ──────────────────────────────────────────────────────────
    # DRL opponents get the first n_drl seats; fixed-policy fill the rest.
    drl_pids = other_pids[:n_drl]
    fixed_pids = other_pids[n_drl:]

    # ── Build DRL opponents ───────────────────────────────────────────────────
    update_drl_opponents = drl_mode == "new"
    drl_opponents: Dict[int, object] = {}

    if drl_mode == "new":
        for pid in drl_pids:
            drl_opponents[pid] = _make_agent(algo, hybrid, pid)
            print(f"  DRL opponent (co-training) → player {pid}")
    else:  # pretrained
        for pid in drl_pids:
            drl_opponents[pid] = _load_agent(algo, hybrid, pid, rival_path)

    # ── Build fixed-policy opponents ──────────────────────────────────────────
    # Cycle through FPAgentA, FPAgentB, FPAgentC for the remaining seats.
    fp_classes = [FPAgentA, FPAgentB, FPAgentC]
    fixed_opponents: Dict[int, object] = {}
    for i, pid in enumerate(fixed_pids):
        cls = fp_classes[i % len(fp_classes)]
        fixed_opponents[pid] = cls(pid)
        print(f"  Fixed-policy opponent ({cls.__name__}) → player {pid}")

    # ── Main agent ────────────────────────────────────────────────────────────
    main_agent = _make_agent(algo, hybrid, main_pid)

    # ── Environment ───────────────────────────────────────────────────────────
    env = MonopolyEnv(agent_ids=[main_pid], max_rounds=200)

    # ── Describe the run ──────────────────────────────────────────────────────
    drl_label = (
        (
            f"{n_drl} co-training DRL"
            if drl_mode == "new"
            else f"{n_drl} pretrained DRL (frozen)"
        )
        if n_drl > 0
        else "0 DRL"
    )
    fixed_label = f"{len(fixed_pids)} fixed-policy"
    mode_str = "Hybrid" if hybrid else "Standard"

    print(f"\n{'=' * 60}")
    print(f"  Algorithm   : {algo.upper()} ({mode_str})")
    print(f"  Opponents   : {drl_label}  +  {fixed_label}")
    if rival_path:
        print(f"  Rival model : {rival_path}")
    print(f"  Games       : {n_games}  |  Log every: {log_every}")
    print(f"{'=' * 60}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = defaultdict(list)
    wins_window = window_games = 0
    window_ti = window_ta = window_td = window_pa = 0

    for game_num in range(1, n_games + 1):
        env.reset()

        result = run_episode(
            env=env,
            main_pid=main_pid,
            main_agent=main_agent,
            drl_opponents=drl_opponents,
            fixed_opponents=fixed_opponents,
            update_drl_opponents=update_drl_opponents,
            is_ppo=is_ppo,
            update_online=True,
        )

        if result["won"]:
            wins_window += 1
        window_games += 1
        window_ti += result["trades_initiated"]
        window_ta += result["trades_accepted"]
        window_td += result["trades_declined"]
        window_pa += result["properties_acquired"]

        if game_num % log_every == 0:
            win_rate = wins_window / window_games * 100
            avg_ti = window_ti / window_games
            avg_ta = window_ta / window_games
            avg_td = window_td / window_games
            avg_pa = window_pa / window_games

            history["win_rates"].append(win_rate)
            history["games"].append(game_num)
            history["rewards"].append(result["reward"])
            history["avg_trades_initiated"].append(avg_ti)
            history["avg_trades_accepted"].append(avg_ta)
            history["avg_trades_declined"].append(avg_td)
            history["avg_properties_acquired"].append(avg_pa)

            eps_str = (
                f"  ε={main_agent.epsilon:.3f}"
                if hasattr(main_agent, "epsilon")
                else ""
            )
            print(
                f"  Game {game_num:5d} | "
                f"Win%: {win_rate:5.1f}%{eps_str} | "
                f"Props: {avg_pa:.1f} | "
                f"Trades init/acc/dec: {avg_ti:.1f}/{avg_ta:.1f}/{avg_td:.1f}"
            )

            wins_window = window_games = 0
            window_ti = window_ta = window_td = window_pa = 0

    return main_agent, dict(history)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train a Monopoly DRL agent against a mix of DRL and fixed opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--algo",
        choices=["ppo", "ddqn"],
        default="ppo",
        help="Algorithm for the main agent and any co-training DRL opponents (default: ppo)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Use hybrid mode — fixed rules for BUY and ACCEPT_TRADE (default: True)",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="hybrid",
        action="store_false",
        help="Disable hybrid mode (pure DRL for all decisions)",
    )
    parser.add_argument(
        "--n-drl",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of DRL opponents (0–3). "
            "Remaining seats are filled with fixed-policy agents. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--drl-mode",
        choices=["new", "pretrained"],
        default="new",
        help=(
            "new        : DRL opponents start fresh and co-train alongside the main agent.\n"
            "pretrained : DRL opponents are loaded from --rival and kept frozen.\n"
            "(default: new)"
        ),
    )
    parser.add_argument(
        "--rival",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a .pt checkpoint used for all pretrained DRL opponents",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2000,
        help="Number of training games (default: 2000)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for saved main-agent weights (default: auto-named)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────────
    if args.n_drl < 0 or args.n_drl > NUM_PLAYERS - 1:
        parser.error(f"--n-drl must be between 0 and {NUM_PLAYERS - 1}")
    if args.drl_mode == "pretrained" and args.rival is None:
        parser.error("--rival PATH is required when --drl-mode pretrained")
    if args.n_drl == 0 and args.drl_mode == "pretrained":
        parser.error("--drl-mode pretrained has no effect when --n-drl 0")

    # ── Auto output name ───────────────────────────────────────────────────────
    if args.out is None:
        mode_tag = "hybrid" if args.hybrid else "standard"
        if args.n_drl == 0:
            opp_tag = "vs_fixed"
        elif args.drl_mode == "new":
            opp_tag = f"vs_{args.n_drl}drl_cotrain"
        else:
            rival_stem = os.path.splitext(os.path.basename(args.rival))[0]
            opp_tag = f"vs_{args.n_drl}drl_{rival_stem}"
        args.out = f"{args.algo}_{mode_tag}_{opp_tag}.pt"

    print(f"\n{'=' * 60}")
    print(f"  Algorithm  : {args.algo.upper()}")
    print(f"  Mode       : {'Hybrid' if args.hybrid else 'Standard'}")
    print(f"  DRL opps   : {args.n_drl} ({args.drl_mode})")
    print(f"  Fixed opps : {NUM_PLAYERS - 1 - args.n_drl}")
    if args.rival:
        print(f"  Rival      : {args.rival}")
    print(f"  Games      : {args.games}")
    print(f"  Save to    : {args.out}")
    print(f"{'=' * 60}")

    start = time.time()

    agent, history = train_mixed(
        algo=args.algo,
        hybrid=args.hybrid,
        n_drl=args.n_drl,
        drl_mode=args.drl_mode,
        rival_path=args.rival,
        n_games=args.games,
        log_every=max(1, args.games // 50),
        seed=args.seed,
    )

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ── Save model ─────────────────────────────────────────────────────────────
    agent.save(args.out)
    print(f"Model saved      : {args.out}")

    # ── Save history ───────────────────────────────────────────────────────────
    history_path = args.out.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved    : {history_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    if history.get("win_rates"):
        final_wr = history["win_rates"][-1]
        best_wr = max(history["win_rates"])
        print(f"\nFinal win rate   : {final_wr:.1f}%")
        print(f"Best win rate    : {best_wr:.1f}%")
    if history.get("avg_properties_acquired"):
        print(f"Avg props/game   : {history['avg_properties_acquired'][-1]:.1f}")
    if history.get("avg_trades_initiated"):
        ti = history["avg_trades_initiated"][-1]
        ta = history["avg_trades_accepted"][-1]
        td = history["avg_trades_declined"][-1]
        print(
            f"Avg trades (last window) — init: {ti:.1f}  acc: {ta:.1f}  dec: {td:.1f}"
        )


if __name__ == "__main__":
    main()
