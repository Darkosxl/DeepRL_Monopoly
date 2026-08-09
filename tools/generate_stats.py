"""
generate_stats.py
-----------------
Simulate N games and collect per-player statistics:
  - Win rate
  - Trades initiated / accepted / declined per game
  - Properties acquired per game
  - Monopolies gained per game

Results are saved to a JSON stats file and (optionally) rendered as plots.

Player slots
------------
Slot 0 is always the "focus" agent — the one whose stats headline the summary.
Slots 1-3 are opponents, each independently configured via --p1 / --p2 / --p3.

Opponent types
--------------
  fixed-a / fixed-b / fixed-c   Fixed-policy agents A, B, C
  fixed                         Alias that cycles A → B → C
  ppo:<path.pt>                 Pre-trained PPO agent (frozen, no learning)
  ddqn:<path.pt>                Pre-trained DDQN agent (frozen, no learning)
  ppo-new                       Fresh (untrained) PPO agent
  ddqn-new                      Fresh (untrained) DDQN agent

Usage examples
--------------
  # Default: focus=PPO hybrid vs FPAgentA, FPAgentB, FPAgentC (4 players)
  python tools/generate_stats.py --model ppo_hybrid_model.pt --games 500

  # 2-player game: focus PPO vs one fixed opponent
  python tools/generate_stats.py --model my.pt --players 2 --p1 fixed-a

  # 3-player game: focus PPO vs pretrained rival + fixed-b
  python tools/generate_stats.py --model my.pt --players 3 --p1 ppo:rival.pt --p2 fixed-b

  # 4-player game: all fixed opponents
  python tools/generate_stats.py --model my.pt --p1 fixed-a --p2 fixed-b --p3 fixed-c

  # DDQN focus agent
  python tools/generate_stats.py --algo ddqn --model ddqn.pt --games 200

  # Save stats to custom path, also generate plots
  python tools/generate_stats.py --model my.pt --games 300 --out results/run1 --plot

  # Skip plots (stats JSON only)
  python tools/generate_stats.py --model my.pt --games 500
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC, FPAgentD, FPAgentE, FPAgentF
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv


# ── Action classification ──────────────────────────────────────────────────────


def _classify_action(a: int) -> Tuple[bool, bool, bool, bool]:
    """(is_buy, is_accept, is_decline, is_trade_offer)"""
    buy = int(ActionType.BUY_PROPERTY)
    acc = int(ActionType.ACCEPT_TRADE)
    dec = int(ActionType.DECLINE_TRADE)
    is_offer = (
        OFFSETS["buy_trade"]  <= a < OFFSETS["buy_trade"]  + 252
        or OFFSETS["sell_trade"] <= a < OFFSETS["sell_trade"] + 252
        or OFFSETS["exch_trade"] <= a < OFFSETS["exch_trade"] + 2268
    )
    return (a == buy, a == acc, a == dec, is_offer)


def _count_monopolies(env, pid: int) -> int:
    """Count how many complete colour monopolies player pid currently holds."""
    count = 0
    for group in COLOR_GROUPS.values():
        if all(
            sq in env.properties and env.properties[sq].owner == pid
            for sq in group
        ):
            count += 1
    return count


# ── Agent factory ──────────────────────────────────────────────────────────────

_FP_CLASSES = {"fixed-a": FPAgentA, "fixed-b": FPAgentB, "fixed-c": FPAgentC, "fixed-d": FPAgentD, "fixed-e": FPAgentE, "fixed-f": FPAgentF}
_FP_CYCLE   = [FPAgentA, FPAgentB, FPAgentC]


def build_opponent(spec: str, player_id: int, fp_cycle_idx: int = 0):
    """
    Parse an opponent spec string and return (agent, label, is_drl).

    spec formats:
      fixed / fixed-a / fixed-b / fixed-c
      ppo:<path>   ddqn:<path>
      ppo-new      ddqn-new
    """
    spec = spec.strip().lower()

    # ── Fixed-policy ──
    if spec == "fixed":
        cls = _FP_CYCLE[fp_cycle_idx % 3]
        return cls(player_id), f"Fixed-{['A','B','C'][fp_cycle_idx % 3]}", False
    if spec in _FP_CLASSES:
        cls = _FP_CLASSES[spec]
        return cls(player_id), f"Fixed-{spec[-1].upper()}", False

    # ── Pre-trained DRL ──
    if spec.startswith("ppo:") or spec.startswith("ddqn:"):
        algo, path = spec.split(":", 1)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rival model not found: {path!r}")
        AgentCls = PPOAgent if algo == "ppo" else DDQNAgent
        agent = AgentCls(player_id=player_id, hybrid=True)
        agent.load(path)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0
        label = f"{algo.upper()}:{os.path.basename(path)}"
        return agent, label, True

    # ── Fresh (untrained) DRL ──
    if spec in ("ppo-new", "ddqn-new"):
        algo = spec.split("-")[0]
        AgentCls = PPOAgent if algo == "ppo" else DDQNAgent
        agent = AgentCls(player_id=player_id, hybrid=True)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0
        return agent, f"{algo.upper()}-new", True

    raise ValueError(
        f"Unknown opponent spec {spec!r}. "
        "Use: fixed / fixed-a / fixed-b / fixed-c / "
        "ppo:<path> / ddqn:<path> / ppo-new / ddqn-new"
    )


# ── Single-game simulation ─────────────────────────────────────────────────────


def run_game(
    env: MonopolyEnv,
    n_players: int,
    focus_pid: int,
    focus_agent,
    focus_is_ppo: bool,
    opponents: Dict[int, object],   # pid → agent (fixed or DRL, never learning)
    opponent_is_drl: Dict[int, bool],
) -> Dict:
    """
    Play one game; return a per-player stats dict.

    Stats tracked for every active player:
        won, trades_initiated, trades_accepted, trades_declined,
        properties_acquired, monopolies_end, steps
    """
    env.reset()

    # Mark unused seats as bankrupt
    for pid in range(n_players, NUM_PLAYERS):
        env.players[pid].bankrupt = True
    env.turn_order = [p for p in env.turn_order if p < n_players]
    env.current_turn_idx = 0

    # Per-player metrics
    metrics = {
        pid: dict(
            trades_initiated=0,
            trades_accepted=0,
            trades_declined=0,
            properties_acquired=0,
            prev_prop_count=len(env.players[pid].properties),
        )
        for pid in range(n_players)
    }
    steps = {pid: 0 for pid in range(n_players)}

    all_agents = {focus_pid: focus_agent}
    all_agents.update(opponents)

    max_steps = env.max_rounds * n_players * 30
    step_count = 0
    done = False

    while not done and step_count < max_steps:
        step_count += 1
        pid = env.whose_turn()

        if env.players[pid].bankrupt:
            env._advance_turn()
            continue

        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.DO_NOTHING)]

        agent = all_agents.get(pid)
        is_drl = (pid == focus_pid) or opponent_is_drl.get(pid, False)

        if is_drl:
            state = env._get_state(pid)
            action = agent.choose_action(state, env, allowed)
            # PPO inference now returns a 4-tuple:
            #   (action, log_prob, value, allowed_actions)
            # while DDQN returns just the action integer.
            # Older call sites in this file expected a 3-tuple, so normalise
            # all DRL agents here by always taking the first tuple element.
            if isinstance(action, tuple):
                action = action[0]
        else:
            # Fixed-policy
            action = agent.choose_action(env)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else allowed[0]
                )

        # Track metrics pre-step
        if pid < n_players:
            is_buy, is_acc, is_dec, is_offer = _classify_action(action)
            m = metrics[pid]
            if is_buy:
                m["properties_acquired"] += 1
            elif is_acc:
                m["trades_accepted"] += 1
            elif is_dec:
                m["trades_declined"] += 1
            elif is_offer:
                m["trades_initiated"] += 1

        _, _, done, _ = env.step(action)

        # Detect trade-based property gain
        if pid < n_players:
            new_count = len(env.players[pid].properties)
            m = metrics[pid]
            if action == int(ActionType.ACCEPT_TRADE) and new_count > m["prev_prop_count"]:
                m["properties_acquired"] += new_count - m["prev_prop_count"]
            m["prev_prop_count"] = new_count
            steps[pid] += 1

    winner = env.winner()

    result = {}
    for pid in range(n_players):
        m = metrics[pid]
        result[pid] = {
            "won":                  int(winner == pid),
            "trades_initiated":     m["trades_initiated"],
            "trades_accepted":      m["trades_accepted"],
            "trades_declined":      m["trades_declined"],
            "properties_acquired":  m["properties_acquired"],
            "monopolies_end":       _count_monopolies(env, pid),
            "steps":                steps[pid],
        }
    return result


# ── Multi-game simulation ──────────────────────────────────────────────────────


def simulate(
    focus_model: Optional[str],
    focus_algo: str,
    focus_hybrid: bool,
    opponent_specs: List[str],   # length = n_players - 1
    n_games: int,
    n_players: int,
    seed: int,
) -> Dict:
    """
    Run n_games games and aggregate per-player statistics.

    Returns a dict ready for JSON serialisation.
    """
    random.seed(seed)
    np.random.seed(seed)

    focus_pid = 0

    # ── Build focus agent ─────────────────────────────────────────────────────
    AgentCls = PPOAgent if focus_algo == "ppo" else DDQNAgent
    focus_agent = AgentCls(player_id=focus_pid, hybrid=focus_hybrid)
    if focus_model and os.path.exists(focus_model):
        focus_agent.load(focus_model)
        print(f"  Focus agent  : {focus_algo.upper()} loaded from {focus_model!r}")
    else:
        print(f"  Focus agent  : {focus_algo.upper()} (untrained weights)")
    if hasattr(focus_agent, "epsilon"):
        focus_agent.epsilon = 0.0
    focus_is_ppo = (focus_algo == "ppo")

    # ── Build opponents ───────────────────────────────────────────────────────
    other_pids = [p for p in range(n_players) if p != focus_pid]
    opponents: Dict[int, object] = {}
    opponent_is_drl: Dict[int, bool] = {}
    opp_labels: Dict[int, str] = {}
    fp_cycle = 0

    for i, pid in enumerate(other_pids):
        spec = opponent_specs[i] if i < len(opponent_specs) else "fixed"
        agent, label, is_drl = build_opponent(spec, pid, fp_cycle)
        if not is_drl:
            fp_cycle += 1
        opponents[pid] = agent
        opponent_is_drl[pid] = is_drl
        opp_labels[pid] = label
        print(f"  Opponent p{pid} : {label}")

    # ── Player name map ───────────────────────────────────────────────────────
    model_stem = (
        os.path.splitext(os.path.basename(focus_model))[0]
        if focus_model
        else f"{focus_algo}-untrained"
    )
    pnames = {focus_pid: f"P0 ({focus_algo.upper()}:{model_stem})"}
    for pid in other_pids:
        pnames[pid] = f"P{pid} ({opp_labels[pid]})"

    # ── Environment ───────────────────────────────────────────────────────────
    env = MonopolyEnv(agent_ids=[focus_pid], max_rounds=200)

    # ── Accumulators ──────────────────────────────────────────────────────────
    accum = {
        pid: defaultdict(list) for pid in range(n_players)
    }

    print(f"\n  Simulating {n_games} games ({n_players} players)...\n")
    t0 = time.time()

    for g in range(1, n_games + 1):
        game_result = run_game(
            env=env,
            n_players=n_players,
            focus_pid=focus_pid,
            focus_agent=focus_agent,
            focus_is_ppo=focus_is_ppo,
            opponents=opponents,
            opponent_is_drl=opponent_is_drl,
        )
        for pid, stats in game_result.items():
            for k, v in stats.items():
                accum[pid][k].append(v)

        if g % max(1, n_games // 10) == 0:
            wr = sum(accum[focus_pid]["won"]) / g * 100
            elapsed = time.time() - t0
            print(f"  [{g:5d}/{n_games}]  focus win rate so far: {wr:5.1f}%  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done — {n_games} games in {elapsed:.1f}s")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _agg(values: list) -> Dict:
        arr = np.array(values, dtype=float)
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
            "raw":    [float(v) for v in values],
        }

    players_out = {}
    for pid in range(n_players):
        a = accum[pid]
        players_out[str(pid)] = {
            "name":                pnames[pid],
            "win_rate_pct":        float(np.mean(a["won"])) * 100,
            "wins":                int(sum(a["won"])),
            "trades_initiated":    _agg(a["trades_initiated"]),
            "trades_accepted":     _agg(a["trades_accepted"]),
            "trades_declined":     _agg(a["trades_declined"]),
            "properties_acquired": _agg(a["properties_acquired"]),
            "monopolies_end":      _agg(a["monopolies_end"]),
            "steps":               _agg(a["steps"]),
        }

    output = {
        "meta": {
            "generated_at":  datetime.now().isoformat(),
            "n_games":       n_games,
            "n_players":     n_players,
            "focus_algo":    focus_algo,
            "focus_hybrid":  focus_hybrid,
            "focus_model":   focus_model,
            "seed":          seed,
            "elapsed_s":     round(elapsed, 2),
            "opponent_specs": opponent_specs,
        },
        "players": players_out,
    }
    return output


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_plots(stats: Dict, out_prefix: str):
    """
    Generate a set of plots from the stats dict and save them to disk.

    Plots produced:
      1. Win rates (bar chart, all players)
      2. Per-game trade activity — initiated / accepted / declined (grouped bar)
      3. Per-game properties acquired (bar chart, all players)
      4. Monopolies at game end (bar chart, all players)
      5. Cumulative win rate over games for the focus player (line chart)
      6. Per-game distribution of trades (box plot, focus player)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ⚠  matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    players = stats["players"]
    pids    = sorted(players.keys(), key=int)
    labels  = [players[p]["name"] for p in pids]
    n_games = stats["meta"]["n_games"]

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else ".", exist_ok=True)

    saved = []

    # ── 1. Win rates ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 2), 5))
    win_rates = [players[p]["win_rate_pct"] for p in pids]
    bars = ax.bar(labels, win_rates, color=[COLORS[i % len(COLORS)] for i in range(len(pids))],
                  edgecolor="white", linewidth=1.2)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{wr:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(100 / len(pids), color="grey", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"Win Rates — {n_games} games")
    ax.set_ylim(0, max(win_rates) * 1.25 + 5)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{out_prefix}_win_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # ── 2. Trade activity (grouped bar) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 3), 5))
    x      = np.arange(len(pids))
    width  = 0.25
    ti     = [players[p]["trades_initiated"]["mean"]  for p in pids]
    ta     = [players[p]["trades_accepted"]["mean"]   for p in pids]
    td     = [players[p]["trades_declined"]["mean"]   for p in pids]
    ti_std = [players[p]["trades_initiated"]["std"]   for p in pids]
    ta_std = [players[p]["trades_accepted"]["std"]    for p in pids]
    td_std = [players[p]["trades_declined"]["std"]    for p in pids]

    ax.bar(x - width, ti, width, label="Initiated", color="#4C72B0",
           yerr=ti_std, capsize=4, edgecolor="white")
    ax.bar(x,          ta, width, label="Accepted",  color="#55A868",
           yerr=ta_std, capsize=4, edgecolor="white")
    ax.bar(x + width,  td, width, label="Declined",  color="#C44E52",
           yerr=td_std, capsize=4, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Avg per game")
    ax.set_title(f"Trade Activity per Game — {n_games} games")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{out_prefix}_trade_activity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # ── 3. Properties acquired per game ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 2), 5))
    prop_means = [players[p]["properties_acquired"]["mean"] for p in pids]
    prop_stds  = [players[p]["properties_acquired"]["std"]  for p in pids]
    ax.bar(labels, prop_means, color=[COLORS[i % len(COLORS)] for i in range(len(pids))],
           yerr=prop_stds, capsize=5, edgecolor="white")
    ax.set_ylabel("Avg properties acquired per game")
    ax.set_title(f"Properties Acquired per Game — {n_games} games")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{out_prefix}_properties.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # ── 4. Monopolies at end of game ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(pids) * 2), 5))
    mono_means = [players[p]["monopolies_end"]["mean"] for p in pids]
    mono_stds  = [players[p]["monopolies_end"]["std"]  for p in pids]
    ax.bar(labels, mono_means, color=[COLORS[i % len(COLORS)] for i in range(len(pids))],
           yerr=mono_stds, capsize=5, edgecolor="white")
    ax.set_ylabel("Avg monopolies at end of game")
    ax.set_title(f"Monopolies at Game End — {n_games} games")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{out_prefix}_monopolies.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # ── 5. Cumulative win rate (focus player) ─────────────────────────────────
    focus_pid_key = "0"
    focus_wins_raw = players[focus_pid_key]["properties_acquired"]["raw"]
    won_raw        = players[focus_pid_key].get("_won_raw")
    # Use the raw wins list stored under a key we can access
    # Re-derive from stats dict if present
    # (we stored raw above; wins are booleans)
    # We need the raw won list — it's stored inside stats at build time.
    # Fallback: re-read from "steps" raw length to infer n
    raw_won = None
    # Check if we can find it — in our _agg we stored "raw" for numeric fields,
    # but "won" is also numeric (0/1). We don't call _agg on it directly, but
    # we do store it as win_rate_pct. We need the raw series.
    # Since we don't expose raw won directly, we skip this plot gracefully.
    # Instead we plot the rolling win rate using properties or steps as proxy.
    # Better: we recompute from the meta. Actually let's store raw won in the
    # JSON output explicitly — see _agg approach below.
    # For now derive cumulative from the data already in stats:
    # We'll store _raw_won in a separate top-level key (added to simulate()).
    raw_won = stats.get("_raw_won_focus")
    if raw_won is not None:
        cum_wr = np.cumsum(raw_won) / (np.arange(len(raw_won)) + 1) * 100
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, len(cum_wr) + 1), cum_wr, color="#4C72B0", linewidth=1.5)
        ax.axhline(100 / len(pids), color="grey", linestyle="--", linewidth=1, label="Random baseline")
        ax.set_xlabel("Games played")
        ax.set_ylabel("Cumulative win rate (%)")
        ax.set_title(f"Focus Agent Cumulative Win Rate — {players[focus_pid_key]['name']}")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = f"{out_prefix}_cumulative_winrate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # ── 6. Box plot: trade distribution for focus player ─────────────────────
    focus = players[focus_pid_key]
    raw_data = {
        "Initiated": focus["trades_initiated"]["raw"],
        "Accepted":  focus["trades_accepted"]["raw"],
        "Declined":  focus["trades_declined"]["raw"],
    }
    if any(raw_data.values()):
        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(
            [raw_data[k] for k in raw_data],
            labels=list(raw_data.keys()),
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        box_colors = ["#4C72B0", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Count per game")
        ax.set_title(f"Trade Distribution — Focus Agent ({players[focus_pid_key]['name']})")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = f"{out_prefix}_trade_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # ── 7. Properties vs Monopolies scatter (focus player) ───────────────────
    raw_props  = focus["properties_acquired"]["raw"]
    raw_monos  = focus["monopolies_end"]["raw"]
    if raw_props and raw_monos:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(raw_props, raw_monos, alpha=0.3, s=20, color="#4C72B0", edgecolors="none")
        # Trend line
        z = np.polyfit(raw_props, raw_monos, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(raw_props), max(raw_props), 100)
        ax.plot(xs, p(xs), color="#DD8452", linewidth=2, label="Trend")
        ax.set_xlabel("Properties acquired")
        ax.set_ylabel("Monopolies at end")
        ax.set_title(f"Properties vs Monopolies — Focus Agent")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = f"{out_prefix}_props_vs_monos.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    print(f"\n  Plots saved:")
    for p in saved:
        print(f"    {p}")
    return saved


# ── Simulate with raw-won injection ───────────────────────────────────────────


def simulate_and_collect(
    focus_model, focus_algo, focus_hybrid,
    opponent_specs, n_games, n_players, seed,
) -> Dict:
    """Wrapper that injects raw won series into the output dict."""
    random.seed(seed)
    np.random.seed(seed)

    focus_pid = 0
    AgentCls  = PPOAgent if focus_algo == "ppo" else DDQNAgent
    focus_agent = AgentCls(player_id=focus_pid, hybrid=focus_hybrid)
    if focus_model and os.path.exists(focus_model):
        focus_agent.load(focus_model)
    if hasattr(focus_agent, "epsilon"):
        focus_agent.epsilon = 0.0
    focus_is_ppo = (focus_algo == "ppo")

    other_pids = [p for p in range(n_players) if p != focus_pid]
    opponents, opponent_is_drl, opp_labels = {}, {}, {}
    fp_cycle = 0
    for i, pid in enumerate(other_pids):
        spec = opponent_specs[i] if i < len(opponent_specs) else "fixed"
        agent, label, is_drl = build_opponent(spec, pid, fp_cycle)
        if not is_drl:
            fp_cycle += 1
        opponents[pid]        = agent
        opponent_is_drl[pid]  = is_drl
        opp_labels[pid]       = label

    focus_model_stem = (
        os.path.splitext(os.path.basename(focus_model))[0]
        if focus_model else f"{focus_algo}-untrained"
    )
    pnames = {focus_pid: f"P0 ({focus_algo.upper()}:{focus_model_stem})"}
    for pid in other_pids:
        pnames[pid] = f"P{pid} ({opp_labels[pid]})"

    env = MonopolyEnv(agent_ids=[focus_pid], max_rounds=200)

    accum    = {pid: defaultdict(list) for pid in range(n_players)}
    raw_won  = []

    print(f"\n{'=' * 60}")
    print(f"  Focus agent  : {focus_algo.upper()} ({'Hybrid' if focus_hybrid else 'Standard'})")
    for pid in other_pids:
        print(f"  Opponent p{pid} : {opp_labels[pid]}")
    print(f"  Games        : {n_games}  |  Players: {n_players}  |  Seed: {seed}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    for g in range(1, n_games + 1):
        game_result = run_game(
            env=env,
            n_players=n_players,
            focus_pid=focus_pid,
            focus_agent=focus_agent,
            focus_is_ppo=focus_is_ppo,
            opponents=opponents,
            opponent_is_drl=opponent_is_drl,
        )
        for pid, s in game_result.items():
            for k, v in s.items():
                accum[pid][k].append(v)
        raw_won.append(game_result[focus_pid]["won"])

        if g % max(1, n_games // 10) == 0:
            wr = sum(raw_won) / g * 100
            elapsed = time.time() - t0
            print(f"  [{g:5d}/{n_games}]  focus win rate: {wr:5.1f}%  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\n  Simulation complete — {n_games} games in {elapsed:.1f}s")

    def _agg(values):
        arr = np.array(values, dtype=float)
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
            "raw":    [float(v) for v in values],
        }

    players_out = {}
    for pid in range(n_players):
        a = accum[pid]
        players_out[str(pid)] = {
            "name":                pnames[pid],
            "win_rate_pct":        float(np.mean(a["won"])) * 100,
            "wins":                int(sum(a["won"])),
            "trades_initiated":    _agg(a["trades_initiated"]),
            "trades_accepted":     _agg(a["trades_accepted"]),
            "trades_declined":     _agg(a["trades_declined"]),
            "properties_acquired": _agg(a["properties_acquired"]),
            "monopolies_end":      _agg(a["monopolies_end"]),
            "steps":               _agg(a["steps"]),
        }

    output = {
        "meta": {
            "generated_at":   datetime.now().isoformat(),
            "n_games":        n_games,
            "n_players":      n_players,
            "focus_algo":     focus_algo,
            "focus_hybrid":   focus_hybrid,
            "focus_model":    focus_model,
            "seed":           seed,
            "elapsed_s":      round(elapsed, 2),
            "opponent_specs": opponent_specs,
        },
        "players":          players_out,
        "_raw_won_focus":   [int(w) for w in raw_won],
    }
    return output


# ── Pretty summary ─────────────────────────────────────────────────────────────


def print_summary(stats: Dict):
    players = stats["players"]
    n_games = stats["meta"]["n_games"]

    print(f"\n{'=' * 60}")
    print(f"  SIMULATION SUMMARY  ({n_games} games)")
    print(f"{'=' * 60}")

    for pid_key in sorted(players.keys(), key=int):
        p = players[pid_key]
        print(f"\n  Player {pid_key}: {p['name']}")
        print(f"  {'─' * 40}")
        print(f"    Win rate            : {p['win_rate_pct']:.1f}%  ({p['wins']}/{n_games})")
        print(f"    Properties/game     : {p['properties_acquired']['mean']:.2f} ± {p['properties_acquired']['std']:.2f}")
        print(f"    Monopolies at end   : {p['monopolies_end']['mean']:.2f} ± {p['monopolies_end']['std']:.2f}")
        print(f"    Trades initiated/gm : {p['trades_initiated']['mean']:.2f} ± {p['trades_initiated']['std']:.2f}")
        print(f"    Trades accepted/gm  : {p['trades_accepted']['mean']:.2f} ± {p['trades_accepted']['std']:.2f}")
        print(f"    Trades declined/gm  : {p['trades_declined']['mean']:.2f} ± {p['trades_declined']['std']:.2f}")

    print(f"\n{'=' * 60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Simulate N Monopoly games and collect per-player statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Focus agent
    parser.add_argument(
        "--model", type=str, default=None, metavar="PATH",
        help="Path to .pt checkpoint for the focus agent (slot 0). "
             "Omit to use untrained weights.",
    )
    parser.add_argument(
        "--algo", choices=["ppo", "ddqn"], default="ppo",
        help="Algorithm for the focus agent (default: ppo)",
    )
    parser.add_argument(
        "--hybrid", action="store_true", default=True,
        help="Use hybrid mode for the focus agent (default: True)",
    )
    parser.add_argument(
        "--no-hybrid", dest="hybrid", action="store_false",
        help="Disable hybrid mode for the focus agent",
    )

    # Game setup
    parser.add_argument(
        "--players", type=int, default=4, choices=[2, 3, 4],
        help="Total number of players including the focus agent (default: 4)",
    )
    parser.add_argument(
        "--games", type=int, default=200,
        help="Number of games to simulate (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Opponents (up to 3, one per non-focus slot)
    parser.add_argument(
        "--p1", type=str, default="fixed",
        metavar="SPEC",
        help="Opponent spec for player slot 1 (default: fixed)",
    )
    parser.add_argument(
        "--p2", type=str, default="fixed",
        metavar="SPEC",
        help="Opponent spec for player slot 2 (default: fixed) — used only if --players ≥ 3",
    )
    parser.add_argument(
        "--p3", type=str, default="fixed",
        metavar="SPEC",
        help="Opponent spec for player slot 3 (default: fixed) — used only if --players = 4",
    )

    # Output
    parser.add_argument(
        "--out", type=str, default=None, metavar="PREFIX",
        help="Output file prefix (no extension). "
             "Stats saved as <PREFIX>_stats.json, plots as <PREFIX>_*.png. "
             "Default: auto-named from model and timestamp.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Generate and save plots (requires matplotlib)",
    )
    parser.add_argument(
        "--no-raw", action="store_true", default=False,
        help="Strip raw per-game series from the JSON output to save space",
    )

    args = parser.parse_args()

    # ── Derive opponent specs based on --players ──────────────────────────────
    n_opponents = args.players - 1
    all_specs   = [args.p1, args.p2, args.p3]
    opp_specs   = all_specs[:n_opponents]

    # ── Auto output prefix ────────────────────────────────────────────────────
    if args.out is None:
        model_tag = (
            os.path.splitext(os.path.basename(args.model))[0]
            if args.model else f"{args.algo}_untrained"
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"stats_{model_tag}_{args.players}p_{ts}"

    # ── Run ───────────────────────────────────────────────────────────────────
    stats = simulate_and_collect(
        focus_model=args.model,
        focus_algo=args.algo,
        focus_hybrid=args.hybrid,
        opponent_specs=opp_specs,
        n_games=args.games,
        n_players=args.players,
        seed=args.seed,
    )

    print_summary(stats)

    # ── Optionally strip raw series before saving ─────────────────────────────
    stats_to_save = stats
    if args.no_raw:
        import copy
        stats_to_save = copy.deepcopy(stats)
        stats_to_save.pop("_raw_won_focus", None)
        for p in stats_to_save["players"].values():
            for field in ("trades_initiated", "trades_accepted", "trades_declined",
                          "properties_acquired", "monopolies_end", "steps"):
                p[field].pop("raw", None)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = f"{args.out}_stats.json"
    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(stats_to_save, f, indent=2)
    print(f"  Stats saved → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        make_plots(stats, args.out)
    else:
        print(
            "\n  Tip: re-run with --plot to generate PNG charts, "
            "or call make_plots() directly on the JSON."
        )

    return stats


# ── Standalone plot regeneration ───────────────────────────────────────────────
# Lets you regenerate plots from a previously saved JSON without re-simulating:
#
#   python tools/generate_stats.py --replot stats_mymodel_4p_20240101_120000_stats.json


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--replot":
        json_path = sys.argv[2]
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            sys.exit(1)
        with open(json_path) as f:
            stored = json.load(f)
        prefix = json_path.replace("_stats.json", "")
        make_plots(stored, prefix)
    else:
        main()
