"""
play_game.py
-------------
Uses the corrected env.py turn structure:
  each iteration asks env.whose_turn() who acts, gets their allowed actions,
  applies one action, and lets the env advance phases internally.

Usage:
    python tools/play_game.py --model ppo_hybrid_model.pt --players 3
    python tools/play_game.py --model ppo_hybrid_model.pt --players 2 --seed 42
    python tools/play_game.py --algo ddqn --model ddqn_model.pt --players 4
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.actions import (
    OFFSETS,
    PROPERTY_IDS,
    ActionType,
    action_to_description,
)
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_game_engine.constants import (
    BOARD,
    COLOR_GROUPS,
    GO_TO_JAIL_SQUARE,
    INCOME_TAX_SQUARE,
    JAIL_BAIL,
    JAIL_SQUARE,
    LUXURY_TAX_SQUARE,
    NUM_PLAYERS,
    PROPERTIES,
    REAL_ESTATE_IDS,
    TRADE_CASH_LEVELS,
)
from monopoly_game_engine.env import (
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    PHASE_PRE_ROLL,
    MonopolyEnv,
    TradeOffer,
)

CHANCE_SQUARES = {7, 22, 36}
COMMUNITY_SQUARES = {2, 17, 33}


# ── Logger ────────────────────────────────────────────────────────────────────


class GameLogger:
    def __init__(self, log_path="game_log.txt"):
        self.log_path = log_path
        self.file = open(log_path, "w", buffering=1)  # line-buffered

    def log(self, text=""):
        print(text)
        self.file.write(text + "\n")

    def separator(self, char="─", width=60):
        self.log(char * width)

    def flush(self):
        self.file.flush()
        self.file.close()
        print(f"\n[Game log saved to: {self.log_path}]")


# ── Name helpers ──────────────────────────────────────────────────────────────


def square_name(sq):
    return BOARD.get(sq, f"Square {sq}")


def _pname_from_pid(pid, env):
    return env._pnames.get(pid, f"Player {pid + 1}")


# ── Action logger ─────────────────────────────────────────────────────────────


def log_action(logger, pid, pname, action_idx, env, info):
    """Log what happened after an action was applied to the env."""
    lines = []

    # Number of cash levels — read from the constant, never hardcoded
    nc = len(TRADE_CASH_LEVELS)

    if action_idx < OFFSETS["mortgage"]:
        atype = ActionType(action_idx)

        if atype == ActionType.ROLL_DICE:
            d1, d2 = env.last_dice
            lines.append(f"{pname} rolls a {d1} and a {d2}  (total: {d1 + d2})")
            sq = env.players[pid].position
            sn = square_name(sq)

            if sq == GO_TO_JAIL_SQUARE:
                lines.append(f"{pname} lands on Go To Jail → sent directly to Jail!")
            elif sq == JAIL_SQUARE and env.players[pid].in_jail:
                lines.append(f"{pname} is in Jail")
            elif sq == INCOME_TAX_SQUARE:
                lines.append(f"{pname} lands on Income Tax — pays $200")
            elif sq == LUXURY_TAX_SQUARE:
                lines.append(f"{pname} lands on Luxury Tax — pays $100")
            elif sq in CHANCE_SQUARES:
                lines.append(f"{pname} lands on Chance (no card effect)")
            elif sq in COMMUNITY_SQUARES:
                lines.append(f"{pname} lands on Community Chest (no card effect)")
            elif sq in env.properties:
                prop = env.properties[sq]
                lines.append(f"{pname} lands on {prop.name}")
                if prop.owner is None:
                    lines.append(f"  → Unowned  |  Price: ${prop.price}")
                elif prop.owner == pid:
                    lines.append(f"  → {pname} owns this property")
                else:
                    rent = info.get("rent_paid", "?")
                    owner_pn = _pname_from_pid(prop.owner, env)
                    lines.append(
                        f"  → Owned by {owner_pn}  |  {pname} pays ${rent} rent"
                    )
            else:
                lines.append(f"{pname} lands on {sn}")

        elif atype == ActionType.BUY_PROPERTY:
            sq = env.players[pid].position
            prop = env.properties.get(sq)
            if prop:
                lines.append(f"{pname} BUYS {prop.name} for ${prop.price}")
                lines.append(f"  → Cash remaining: ${env.players[pid].cash}")

        elif atype == ActionType.END_TURN:
            lines.append(f"{pname} ends their turn")

        elif atype == ActionType.DO_NOTHING:
            pass

        elif atype == ActionType.USE_GOOJ_CARD:
            lines.append(f"{pname} uses Get Out of Jail Free card — released!")

        elif atype == ActionType.PAY_BAIL:
            lines.append(f"{pname} pays ${JAIL_BAIL} bail — released from Jail")

        elif atype == ActionType.ACCEPT_TRADE:
            lines.append(f"{pname} ACCEPTS the trade offer")

        elif atype == ActionType.DECLINE_TRADE:
            lines.append(f"{pname} DECLINES the trade offer")

        elif atype == ActionType.DECLARE_BANKRUPT:
            lines.append(f"💀 {pname} declares BANKRUPTCY!")

    elif action_idx < OFFSETS["unmortgage"]:
        local = action_idx - OFFSETS["mortgage"]
        prop = env.properties[PROPERTY_IDS[local]]
        lines.append(f"{pname} mortgages {prop.name} — receives ${prop.mortgage_v}")

    elif action_idx < OFFSETS["improve_house"]:
        local = action_idx - OFFSETS["unmortgage"]
        prop = env.properties[PROPERTY_IDS[local]]
        cost = int(prop.mortgage_v * 1.1)
        lines.append(f"{pname} lifts mortgage on {prop.name} — pays ${cost}")

    elif action_idx < OFFSETS["improve_hotel"]:
        local = action_idx - OFFSETS["improve_house"]
        prop = env.properties[REAL_ESTATE_IDS[local]]
        lines.append(f"{pname} builds a HOUSE on {prop.name}  ({prop.houses} house(s))")

    elif action_idx < OFFSETS["sell_house"]:
        local = action_idx - OFFSETS["improve_hotel"]
        prop = env.properties[REAL_ESTATE_IDS[local]]
        lines.append(f"{pname} builds a HOTEL on {prop.name}!")

    elif action_idx < OFFSETS["sell_hotel"]:
        local = action_idx - OFFSETS["sell_house"]
        prop = env.properties[REAL_ESTATE_IDS[local]]
        lines.append(f"{pname} sells a house on {prop.name}")

    elif action_idx < OFFSETS["sell_prop"]:
        local = action_idx - OFFSETS["sell_hotel"]
        prop = env.properties[REAL_ESTATE_IDS[local]]
        lines.append(f"{pname} sells the hotel on {prop.name}")

    elif action_idx < OFFSETS["buy_trade"]:
        local = action_idx - OFFSETS["sell_prop"]
        prop = env.properties[PROPERTY_IDS[local]]
        lines.append(f"{pname} sells {prop.name} back to bank for ${prop.mortgage_v}")

    elif action_idx < OFFSETS["sell_trade"]:
        local = action_idx - OFFSETS["buy_trade"]
        n = len(PROPERTY_IDS)
        t_idx = local // (n * nc)
        rem = local % (n * nc)
        prop = env.properties[PROPERTY_IDS[rem // nc]]
        price_lvl = TRADE_CASH_LEVELS[rem % nc]
        others = [i for i in range(NUM_PLAYERS) if i != pid]
        target_pid = others[t_idx] if t_idx < len(others) else others[0]
        cash = int(prop.price * price_lvl)
        target_pn = _pname_from_pid(target_pid, env)
        lines.append(f"{pname} sends a BUY offer to {target_pn}:")
        lines.append(f"  ► Wants: {prop.name}  |  Offering: ${cash}")

    elif action_idx < OFFSETS["exch_trade"]:
        local = action_idx - OFFSETS["sell_trade"]
        n = len(PROPERTY_IDS)
        t_idx = local // (n * nc)
        rem = local % (n * nc)
        prop = env.properties[PROPERTY_IDS[rem // nc]]
        price_lvl = TRADE_CASH_LEVELS[rem % nc]
        others = [i for i in range(NUM_PLAYERS) if i != pid]
        target_pid = others[t_idx] if t_idx < len(others) else others[0]
        cash = int(prop.price * price_lvl)
        target_pn = _pname_from_pid(target_pid, env)
        lines.append(f"{pname} sends a SELL offer to {target_pn}:")
        lines.append(f"  ► Offering: {prop.name}  |  Requesting: ${cash}")

    elif action_idx < OFFSETS["auction"]:
        local = action_idx - OFFSETS["exch_trade"]
        n = len(PROPERTY_IDS)
        t_idx = local // (n * (n - 1))
        rem = local % (n * (n - 1))
        oi = rem // (n - 1)
        ri_raw = rem % (n - 1)
        ri = ri_raw if ri_raw < oi else ri_raw + 1
        others = [i for i in range(NUM_PLAYERS) if i != pid]
        target_pid = others[t_idx] if t_idx < len(others) else others[0]
        offered = env.properties[PROPERTY_IDS[oi]]
        requested = env.properties[PROPERTY_IDS[ri]]
        target_pn = _pname_from_pid(target_pid, env)
        lines.append(f"{pname} sends an EXCHANGE offer to {target_pn}:")
        lines.append(f"  ► Offering:   {offered.name}  (${offered.price})")
        lines.append(f"  ► Requesting: {requested.name}  (${requested.price})")

    else:
        lines.append(f"{pname}: {action_to_description(action_idx)}")

    if "auction_winner" in info:
        winner = _pname_from_pid(info["auction_winner"], env)
        lines.append(f"  → {winner} wins for ${info['auction_price']}")

    for line in lines:
        if line:
            logger.log(f"  {line}")


# ── Standings snapshot ────────────────────────────────────────────────────────


def log_standings(logger, env, n_players):
    logger.log()
    logger.log("  📊 Current standings:")
    for pid in range(n_players):
        player = env.players[pid]
        pname = env._pnames[pid]
        if player.bankrupt:
            logger.log(f"    {pname}: BANKRUPT")
            continue
        logger.log(
            f"    {pname}: "
            f"Cash=${player.cash}  |  "
            f"Properties={len(player.properties)}  |  "
            f"Monopolies={player.num_monopolies()}  |  "
            f"Net Worth=${player.net_worth():.0f}"
        )
    logger.log()


# ── Main simulation ───────────────────────────────────────────────────────────


def simulate(model_path, algo, n_players, log_path):
    logger = GameLogger(log_path)

    logger.separator("═")
    logger.log("  MONOPOLY — AI Game Simulation")
    logger.log(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"  Model  : {model_path}")
    logger.log(f"  Players: {n_players}")
    logger.separator("═")

    # ── Load trained agent ────────────────────────────────────────────────
    trained_pid = 0
    if algo == "ppo":
        trained_agent = PPOAgent(player_id=trained_pid, hybrid=True)
    else:
        trained_agent = DDQNAgent(player_id=trained_pid, hybrid=True)

    if model_path and os.path.exists(model_path):
        trained_agent.load(model_path)
        logger.log(f"\n✓ Loaded trained model: {model_path}")
    else:
        logger.log(f"\n⚠ No model at '{model_path}' — using untrained weights")

    if hasattr(trained_agent, "epsilon"):
        trained_agent.epsilon = 0.0

    # ── Fixed-policy opponents ────────────────────────────────────────────
    fp_classes = [FPAgentA, FPAgentB, FPAgentC]
    other_pids = list(range(1, n_players))
    fp_agents = {
        other_pids[i]: fp_classes[i % 3](other_pids[i]) for i in range(len(other_pids))
    }

    pnames = {trained_pid: "Player 1 (AI★)"}
    for pid in other_pids:
        pnames[pid] = f"Player {pid + 1}"

    agents_map = {trained_pid: trained_agent}
    agents_map.update(fp_agents)

    # ── Roster ────────────────────────────────────────────────────────────
    logger.log("\n  Players:")
    for pid in range(n_players):
        role = (
            "Trained model"
            if pid == trained_pid
            else f"Fixed-policy (FP-{['A', 'B', 'C'][pid - 1]})"
        )
        logger.log(f"    {pnames[pid]}  →  {role}")

    # ── Build env ─────────────────────────────────────────────────────────
    env = MonopolyEnv(agent_ids=[trained_pid], max_rounds=200)
    env.reset()

    # Attach name maps to env for logging helpers
    env._pnames = pnames
    env._pnames_rev = {v: k for k, v in pnames.items()}

    # Mark unused player slots as bankrupt
    for pid in range(n_players, NUM_PLAYERS):
        env.players[pid].bankrupt = True

    # Restrict turn order to active players only
    env.turn_order = [p for p in env.turn_order if p < n_players]
    env.current_turn_idx = 0

    logger.separator()
    logger.log(f"\n  Turn order: {' → '.join(pnames[p] for p in env.turn_order)}")
    logger.separator()

    # ── Game loop ─────────────────────────────────────────────────────────
    current_round = -1
    step_limit = 10_000
    steps = 0
    bankrupt_announced = set()

    while not env.done and steps < step_limit:
        steps += 1

        # Round header
        if env.round != current_round:
            current_round = env.round
            logger.separator()
            logger.log(f"  ROUND {current_round + 1}")
            logger.separator()
            log_standings(logger, env, n_players)

        # Who acts right now?
        pid = env.whose_turn()
        pname = pnames.get(pid, f"Player {pid + 1}")

        # Skip bankrupt players
        if env.players[pid].bankrupt:
            if pid not in bankrupt_announced:
                bankrupt_announced.add(pid)
                logger.log(f"  💀 {pname} is bankrupt and eliminated")
            env._advance_turn()
            continue

        # Get allowed actions
        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.END_TURN)]

        # Ask agent for action
        if pid == trained_pid:
            state = env._get_state(pid)
            if algo == "ppo":
                action, _, _, _ = trained_agent.choose_action(state, env, allowed)
            else:
                action = trained_agent.choose_action(state, env, allowed)
        else:
            agent = agents_map[pid]
            action = agent.choose_action(env)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else allowed[0]
                )

        # Apply to env
        _, _, done, info = env.step(action)

        # Log (skip silent DO_NOTHING)
        if action != int(ActionType.DO_NOTHING):
            log_action(logger, pid, pname, action_idx=action, env=env, info=info)

        # Announce any newly bankrupt players
        for p in range(n_players):
            if env.players[p].bankrupt and p not in bankrupt_announced:
                bankrupt_announced.add(p)
                logger.log(f"  💀 {pnames[p]} has gone BANKRUPT and is eliminated!")

    # ── Game over ─────────────────────────────────────────────────────────
    logger.log()
    logger.separator("═")
    logger.log("  GAME OVER")
    logger.separator("═")

    winner = env.winner()
    if winner is not None and winner < n_players:
        logger.log(f"\n  🏆 WINNER: {pnames[winner]}!")
    else:
        logger.log("\n  Round limit reached")
        richest = max(range(n_players), key=lambda p: env.players[p].net_worth())
        logger.log(f"  🏆 WINNER by net worth: {pnames[richest]}!")

    logger.log()
    logger.log("  Final standings:")
    for pid in range(n_players):
        player = env.players[pid]
        status = (
            "BANKRUPT" if player.bankrupt else f"Net Worth: ${player.net_worth():.0f}"
        )
        props = [p.name for p in player.properties]
        logger.log(f"    {pnames[pid]}: {status}")
        if props:
            logger.log(f"      Properties: {', '.join(props)}")

    logger.separator("═")
    logger.flush()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_hybrid_model.pt")
    parser.add_argument("--algo", choices=["ppo", "ddqn"], default="ppo")
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--log", type=str, default="game_log.txt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not 2 <= args.players <= 4:
        print("Error: --players must be 2, 3, or 4")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    simulate(
        model_path=args.model,
        algo=args.algo,
        n_players=args.players,
        log_path=args.log,
    )
