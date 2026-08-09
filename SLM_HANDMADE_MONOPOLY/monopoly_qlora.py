"""Shared contracts for the Gemma Monopoly teacher, dataset, and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = PROJECT_ROOT / "RL_PPO(UNOFFICIAL)_MONOPOLY"
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))

from monopoly_drl.actions import (  # noqa: E402
    ACTION_SPACE_SIZE,
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    ActionType,
    AuctionAction,
)
from monopoly_drl.agent_ppo import (  # noqa: E402
    fixed_accept_trade_decision,
    fixed_buy_decision,
)
from monopoly_drl.agents_fixed import FP_AGENT_CLASSES  # noqa: E402
from monopoly_drl.constants import (  # noqa: E402
    NUM_PLAYERS,
    RULESET_VERSION,
    TRADE_CASH_LEVELS,
)
from monopoly_drl.env import MonopolyEnv  # noqa: E402
from monopoly_drl.train import run_episode  # noqa: E402


SCHEMA_VERSION = "monopoly-decision-v1"
PRICE_PCTS = tuple(round(level * 100) for level in TRADE_CASH_LEVELS)
PROPERTY_ACTIONS = {
    "mortgage": ("mortgage", PROPERTY_IDS),
    "unmortgage": ("unmortgage", PROPERTY_IDS),
    "improve_house": ("improve_house", REAL_ESTATE_IDS),
    "improve_hotel": ("improve_hotel", REAL_ESTATE_IDS),
    "sell_house": ("sell_house", REAL_ESTATE_IDS),
    "sell_hotel": ("sell_hotel", REAL_ESTATE_IDS),
    "sell_property": ("sell_prop", PROPERTY_IDS),
}
SYSTEM_PROMPT = (
    "You are SELF in a four-player Monopoly game. Choose exactly one legal "
    "action. Return one JSON object only: no rationale, markdown, or code fence."
)


class DecisionFormatError(ValueError):
    """Model output is not one exact legal Monopoly action object."""


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seat_order(env: MonopolyEnv, actor_pid: int) -> list[int]:
    """Return physical seats in canonical SELF, OPP1, OPP2, OPP3 turn order."""
    if actor_pid not in env.turn_order:
        raise ValueError(f"Actor {actor_pid} is absent from turn order")
    start = env.turn_order.index(actor_pid)
    return [env.turn_order[(start + offset) % NUM_PLAYERS] for offset in range(NUM_PLAYERS)]


def seat_names(env: MonopolyEnv, actor_pid: int) -> dict[int, str]:
    return {
        pid: "SELF" if index == 0 else f"OPP{index}"
        for index, pid in enumerate(seat_order(env, actor_pid))
    }


def _section(action: int) -> str:
    names = sorted(OFFSETS, key=OFFSETS.get)
    for index, name in enumerate(names):
        end = OFFSETS[names[index + 1]] if index + 1 < len(names) else ACTION_SPACE_SIZE
        if OFFSETS[name] <= action < end:
            return name
    raise DecisionFormatError(f"Action id outside action space: {action}")


def action_family(action: int) -> str:
    section = _section(action)
    if section == "binary":
        return ActionType(action).name.lower()
    if section == "auction":
        return "auction"
    return section


def action_to_object(action: int, env: MonopolyEnv, actor_pid: int) -> dict:
    """Convert any simulator action id to its canonical JSON object."""
    if not 0 <= int(action) < ACTION_SPACE_SIZE:
        raise DecisionFormatError(f"Action id outside action space: {action}")
    action = int(action)
    section = _section(action)

    if section == "binary":
        return {"action": ActionType(action).name.lower()}
    if section == "auction":
        if action == int(AuctionAction.PASS):
            return {"action": "auction_pass"}
        try:
            amount = AUCTION_ACTION_TO_INCREMENT[AuctionAction(action)]
        except (ValueError, KeyError) as exc:
            raise DecisionFormatError(f"Unknown auction action: {action}") from exc
        return {"action": "auction_bid", "amount": amount}

    for output_name, (offset_name, squares) in PROPERTY_ACTIONS.items():
        if section == offset_name:
            return {
                "action": output_name,
                "square": squares[action - OFFSETS[offset_name]],
            }

    names = seat_names(env, actor_pid)
    n_props = len(PROPERTY_IDS)
    if section in ("buy_trade", "sell_trade"):
        local = action - OFFSETS[section]
        target_slot, rem = divmod(local, n_props * len(PRICE_PCTS))
        prop_slot, price_slot = divmod(rem, len(PRICE_PCTS))
        target_pid = [pid for pid in range(NUM_PLAYERS) if pid != actor_pid][target_slot]
        return {
            "action": section,
            "target": names[target_pid],
            "square": PROPERTY_IDS[prop_slot],
            "price_pct": PRICE_PCTS[price_slot],
        }
    if section == "exch_trade":
        local = action - OFFSETS[section]
        target_slot, rem = divmod(local, n_props * (n_props - 1))
        offer_slot, request_raw = divmod(rem, n_props - 1)
        request_slot = request_raw if request_raw < offer_slot else request_raw + 1
        target_pid = [pid for pid in range(NUM_PLAYERS) if pid != actor_pid][target_slot]
        return {
            "action": "exchange_trade",
            "target": names[target_pid],
            "offer_square": PROPERTY_IDS[offer_slot],
            "request_square": PROPERTY_IDS[request_slot],
        }
    raise DecisionFormatError(f"Unsupported action section: {section}")


def action_to_json(action: int, env: MonopolyEnv, actor_pid: int) -> str:
    return json.dumps(action_to_object(action, env, actor_pid), separators=(",", ":"))


def _exact_keys(value: dict, required: set[str]) -> None:
    if set(value) != required:
        raise DecisionFormatError(
            f"Expected keys {sorted(required)}, received {sorted(value)}"
        )


def object_to_action(
    value: dict,
    env: MonopolyEnv,
    actor_pid: int,
    legal_actions: Sequence[int] | None = None,
) -> int:
    """Convert a strict canonical object to an id, optionally requiring legality."""
    if not isinstance(value, dict) or type(value.get("action")) is not str:
        raise DecisionFormatError("Output must be one JSON object with a string action")
    name = value["action"]
    binary = {member.name.lower(): int(member) for member in ActionType}

    if name in binary:
        _exact_keys(value, {"action"})
        action = binary[name]
    elif name == "auction_pass":
        _exact_keys(value, {"action"})
        action = int(AuctionAction.PASS)
    elif name == "auction_bid":
        _exact_keys(value, {"action", "amount"})
        if type(value["amount"]) is not int:
            raise DecisionFormatError("Auction amount must be an integer")
        reverse = {amount: int(action) for action, amount in AUCTION_ACTION_TO_INCREMENT.items()}
        if value["amount"] not in reverse:
            raise DecisionFormatError(f"Unsupported auction amount: {value['amount']}")
        action = reverse[value["amount"]]
    elif name in PROPERTY_ACTIONS:
        _exact_keys(value, {"action", "square"})
        offset_name, squares = PROPERTY_ACTIONS[name]
        if type(value["square"]) is not int or value["square"] not in squares:
            raise DecisionFormatError(f"Invalid square for {name}: {value['square']}")
        action = OFFSETS[offset_name] + squares.index(value["square"])
    elif name in ("buy_trade", "sell_trade"):
        _exact_keys(value, {"action", "target", "square", "price_pct"})
        action = _cash_trade_to_action(value, env, actor_pid)
    elif name == "exchange_trade":
        _exact_keys(
            value,
            {"action", "target", "offer_square", "request_square"},
        )
        action = _exchange_to_action(value, env, actor_pid)
    else:
        raise DecisionFormatError(f"Unknown action: {name}")

    if legal_actions is not None and action not in legal_actions:
        raise DecisionFormatError(f"Action {action} is not legal in this state")
    return action


def _target_pid(value: dict, env: MonopolyEnv, actor_pid: int) -> int:
    reverse = {name: pid for pid, name in seat_names(env, actor_pid).items()}
    target = value.get("target")
    if type(target) is not str or target == "SELF" or target not in reverse:
        raise DecisionFormatError(f"Invalid trade target: {target}")
    return reverse[target]


def _cash_trade_to_action(value: dict, env: MonopolyEnv, actor_pid: int) -> int:
    target_pid = _target_pid(value, env, actor_pid)
    square, price_pct = value.get("square"), value.get("price_pct")
    if type(square) is not int or square not in PROPERTY_IDS:
        raise DecisionFormatError(f"Invalid trade square: {square}")
    if type(price_pct) is not int or price_pct not in PRICE_PCTS:
        raise DecisionFormatError(f"Invalid price percentage: {price_pct}")
    others = [pid for pid in range(NUM_PLAYERS) if pid != actor_pid]
    stride = len(PROPERTY_IDS) * len(PRICE_PCTS)
    return (
        OFFSETS[value["action"]]
        + others.index(target_pid) * stride
        + PROPERTY_IDS.index(square) * len(PRICE_PCTS)
        + PRICE_PCTS.index(price_pct)
    )


def _exchange_to_action(value: dict, env: MonopolyEnv, actor_pid: int) -> int:
    target_pid = _target_pid(value, env, actor_pid)
    offer, request = value.get("offer_square"), value.get("request_square")
    if type(offer) is not int or offer not in PROPERTY_IDS:
        raise DecisionFormatError(f"Invalid offered square: {offer}")
    if type(request) is not int or request not in PROPERTY_IDS or request == offer:
        raise DecisionFormatError(f"Invalid requested square: {request}")
    others = [pid for pid in range(NUM_PLAYERS) if pid != actor_pid]
    n_props = len(PROPERTY_IDS)
    offer_slot = PROPERTY_IDS.index(offer)
    request_slot = PROPERTY_IDS.index(request)
    request_raw = request_slot if request_slot < offer_slot else request_slot - 1
    return (
        OFFSETS["exch_trade"]
        + others.index(target_pid) * n_props * (n_props - 1)
        + offer_slot * (n_props - 1)
        + request_raw
    )


def parse_action_json(
    raw: str,
    env: MonopolyEnv,
    actor_pid: int,
    legal_actions: Sequence[int] | None = None,
) -> int:
    if type(raw) is not str or "```" in raw:
        raise DecisionFormatError("Output must be plain JSON without a code fence")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionFormatError(f"Invalid JSON: {exc.msg}") from exc
    return object_to_action(value, env, actor_pid, legal_actions)


def fallback_action(legal_actions: Sequence[int]) -> int:
    if not legal_actions:
        raise ValueError("No legal actions are available")
    priorities = (
        int(ActionType.END_TURN),
        int(ActionType.ROLL_DICE),
        int(AuctionAction.PASS),
        int(ActionType.DECLINE_TRADE),
    )
    return next((action for action in priorities if action in legal_actions), legal_actions[0])


def parse_or_fallback(raw: str, env: MonopolyEnv, actor_pid: int) -> tuple[int, str | None]:
    legal = env.get_allowed_actions(actor_pid)
    try:
        return parse_action_json(raw, env, actor_pid, legal), None
    except DecisionFormatError as exc:
        return fallback_action(legal), str(exc)


def grouped_legal_actions(
    env: MonopolyEnv, actor_pid: int, legal_actions: Sequence[int] | None = None
) -> dict:
    """Compact expanded simulator ids into model-facing legal domains."""
    legal = list(env.get_allowed_actions(actor_pid) if legal_actions is None else legal_actions)
    objects = [action_to_object(action, env, actor_pid) for action in legal]
    grouped: dict[str, object] = {}
    exchange: dict[str, dict[str, set[int]]] = {}
    cash_trades: dict[str, dict[str, dict[int, set[int]]]] = {
        "buy_trade": {},
        "sell_trade": {},
    }

    for value in objects:
        name = value["action"]
        if name in {member.name.lower() for member in ActionType} or name == "auction_pass":
            grouped.setdefault("binary", []).append(name)
        elif name == "auction_bid":
            grouped.setdefault("auction_bid", []).append(value["amount"])
        elif name in PROPERTY_ACTIONS:
            grouped.setdefault(name, []).append(value["square"])
        elif name in ("buy_trade", "sell_trade"):
            target = cash_trades[name].setdefault(value["target"], {})
            target.setdefault(value["square"], set()).add(value["price_pct"])
        elif name == "exchange_trade":
            domain = exchange.setdefault(
                value["target"], {"offer": set(), "request": set()}
            )
            domain["offer"].add(value["offer_square"])
            domain["request"].add(value["request_square"])

    for name, targets in cash_trades.items():
        if not targets:
            continue
        grouped[name] = {}
        for target_name, squares in sorted(targets.items()):
            price_domains: dict[str, list[int]] = defaultdict(list)
            for square, prices in sorted(squares.items()):
                price_domains[",".join(map(str, sorted(prices)))].append(square)
            grouped[name][target_name] = dict(sorted(price_domains.items()))
    if exchange:
        grouped["exchange_trade"] = {
            target: {key: sorted(squares) for key, squares in domain.items()}
            for target, domain in sorted(exchange.items())
        }
    return grouped


def canonical_state(env: MonopolyEnv, actor_pid: int) -> dict:
    names = seat_names(env, actor_pid)
    players = []
    for pid in seat_order(env, actor_pid):
        player = env.players[pid]
        players.append(
            [
                names[pid],
                player.position,
                player.cash,
                round(player.net_worth()),
                int(player.in_jail),
                player.jail_turns,
                int(player.gooj_card),
                int(player.bankrupt),
            ]
        )
    deeds = [
        [square, names[prop.owner], int(prop.mortgaged), prop.houses]
        for square, prop in sorted(env.properties.items())
        if prop.owner is not None
    ]
    incoming = env._incoming_trade(actor_pid)
    trade = None
    if incoming is not None:
        trade = {
            "from": names[incoming.from_player],
            "offer": None if incoming.offered_prop is None else incoming.offered_prop.square_id,
            "request": None if incoming.requested_prop is None else incoming.requested_prop.square_id,
            "cash_offer": incoming.cash_offered,
            "cash_request": incoming.cash_requested,
        }
    auction = None
    if env.auction_property_id is not None:
        auction = {
            "square": env.auction_property_id,
            "high_bid": env.auction_high_bid,
            "leader": None if env.auction_high_bidder is None else names[env.auction_high_bidder],
            "bidders": [names[pid] for pid in env.auction_bidders],
        }
    return {
        "ruleset": RULESET_VERSION,
        "phase": env.phase,
        "round": env.round,
        "acting": names[env.whose_turn()],
        "active": names[env.active_player_id()],
        "rolled": int(env.has_rolled),
        "dice": list(env.last_dice),
        "supply": [env.houses_available, env.hotels_available],
        "debt": [env.debt_amount, None if env.debt_creditor is None else names[env.debt_creditor]],
        "players": players,
        "deeds": deeds,
        "trade": trade,
        "auction": auction,
    }


def serialize_decision(env: MonopolyEnv, actor_pid: int) -> str:
    payload = canonical_state(env, actor_pid)
    payload["legal"] = grouped_legal_actions(env, actor_pid)
    return canonical_json(payload)


def canonical_prompt(env: MonopolyEnv, actor_pid: int) -> str:
    return f"{SYSTEM_PROMPT}\n{serialize_decision(env, actor_pid)}"


def deterministic_ppo_action(agent, env: MonopolyEnv, actor_pid: int) -> tuple[int, dict[int, float]]:
    """Argmax PPO decision and legal probabilities, independent of physical seat id."""
    legal = env.get_allowed_actions(actor_pid)
    hybrid_action = None
    if getattr(agent, "hybrid", False):
        if int(ActionType.BUY_PROPERTY) in legal and fixed_buy_decision(env, actor_pid):
            hybrid_action = int(ActionType.BUY_PROPERTY)
        elif int(ActionType.ACCEPT_TRADE) in legal:
            hybrid_action = (
                int(ActionType.ACCEPT_TRADE)
                if fixed_accept_trade_decision(env, actor_pid)
                else int(ActionType.DECLINE_TRADE)
            )
        neural_legal = [action for action in legal if not agent.fixed_action_mask[action]]
    else:
        neural_legal = legal
    if not neural_legal:
        return hybrid_action, {hybrid_action: 1.0}
    state = env._get_state(actor_pid)
    device = next(agent.actor.parameters()).device
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.zeros((1, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
    mask[0, neural_legal] = True
    with torch.inference_mode():
        probabilities = agent.actor(state_tensor, mask).exp().squeeze(0)
    result = {action: float(probabilities[action]) for action in neural_legal}
    if hybrid_action is not None:
        result[hybrid_action] = 1.0
        return hybrid_action, result
    return max(neural_legal, key=lambda action: (result[action], -action)), result


def shortlist_actions(
    legal_actions: Sequence[int],
    teacher_action: int,
    probabilities: Mapping[int, float],
    limit: int = 16,
) -> list[int]:
    if teacher_action not in legal_actions:
        raise ValueError("Teacher action is not legal")
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    legal = list(dict.fromkeys(int(action) for action in legal_actions))
    selected = [teacher_action]
    mandatory = {
        int(ActionType.ROLL_DICE),
        int(ActionType.BUY_PROPERTY),
        int(ActionType.ACCEPT_TRADE),
        int(ActionType.DECLINE_TRADE),
        int(ActionType.DECLARE_BANKRUPT),
        int(AuctionAction.PASS),
    }
    selected.extend(action for action in legal if action in mandatory and action not in selected)
    by_family: dict[str, list[int]] = defaultdict(list)
    for action in legal:
        by_family[action_family(action)].append(action)
    for family in sorted(by_family):
        best = max(
            by_family[family],
            key=lambda action: (probabilities.get(action, 0.0), -action),
        )
        if best not in selected:
            selected.append(best)
    remaining = sorted(
        (action for action in legal if action not in selected),
        key=lambda action: (-probabilities.get(action, 0.0), action),
    )
    return (selected + remaining)[:limit]


def rollout_value(
    env: MonopolyEnv,
    first_action: int,
    actor_pid: int,
    policy: Callable[[MonopolyEnv, int], int],
    seed: int,
    horizon: int = 256,
) -> float:
    simulation = copy.deepcopy(env)
    random.seed(seed)
    np.random.seed(seed)
    if first_action not in simulation.get_allowed_actions(actor_pid):
        raise ValueError(f"Rollout candidate {first_action} is illegal")
    simulation.step(first_action)
    decisions = 1
    while not simulation.done and decisions < horizon:
        pid = simulation.whose_turn()
        legal = simulation.get_allowed_actions(pid)
        action = policy(simulation, pid)
        if action not in legal:
            action = fallback_action(legal)
        simulation.step(action)
        decisions += 1
    if simulation.done:
        return 1.0 if simulation.winner() == actor_pid else -1.0
    return simulation._compute_reward(actor_pid)


def score_candidates(
    env: MonopolyEnv,
    candidates: Sequence[int],
    actor_pid: int,
    policy: Callable[[MonopolyEnv, int], int],
    seeds: Sequence[int],
    horizon: int = 256,
) -> dict[int, list[float]]:
    return {
        action: [rollout_value(env, action, actor_pid, policy, seed, horizon) for seed in seeds]
        for action in candidates
    }


def best_relabel(scores: Mapping[int, Sequence[float]], min_margin: float = 0.05):
    if not scores:
        raise ValueError("No rollout scores supplied")
    ranked = sorted(
        ((float(np.mean(values)), int(action)) for action, values in scores.items()),
        key=lambda item: (-item[0], item[1]),
    )
    margin = math.inf if len(ranked) == 1 else ranked[0][0] - ranked[1][0]
    return (ranked[0][1] if margin >= min_margin else None), margin


def make_dataset_row(
    *,
    env: MonopolyEnv,
    actor_pid: int,
    game_id: str,
    seed: int,
    step: int,
    teacher_action: int,
    relabeled_action: int,
    rollout_scores: Mapping[int, Sequence[float]],
    outcome: str,
    teacher_checkpoint_hash: str,
) -> dict:
    legal = env.get_allowed_actions(actor_pid)
    if teacher_action not in legal or relabeled_action not in legal:
        raise ValueError("Dataset actions must be legal in the recorded state")
    prompt = canonical_prompt(env, actor_pid)
    completion = action_to_json(relabeled_action, env, actor_pid)
    return {
        "schema": SCHEMA_VERSION,
        "ruleset": RULESET_VERSION,
        "game_id": str(game_id),
        "seed": int(seed),
        "step": int(step),
        "actor_pid": int(actor_pid),
        "seat_order": seat_order(env, actor_pid),
        "phase": env.phase,
        "action_family": action_family(relabeled_action),
        "prompt": prompt,
        "completion": completion,
        "teacher_action": int(teacher_action),
        "relabeled_action": int(relabeled_action),
        "legal_actions": sorted(int(action) for action in legal),
        "rollout_scores": {
            str(action): [float(score) for score in values]
            for action, values in sorted(rollout_scores.items())
        },
        "outcome": outcome,
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
        "state_hash": sha256_text(prompt),
    }


def collect_relabelled_game(
    *,
    env: MonopolyEnv,
    teacher,
    teacher_pid: int,
    opponents,
    game_id: str,
    seed: int,
    teacher_checkpoint_hash: str,
    rollouts_per_action: int = 4,
    rollout_horizon: int = 256,
    candidate_limit: int = 16,
    min_margin: float = 0.05,
) -> tuple[list[dict], dict]:
    """Collect non-forced teacher states and relabel them with CRN rollouts."""
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    opponent_map = {agent.player_id: agent for agent in opponents}

    def rollout_policy(simulation: MonopolyEnv, pid: int) -> int:
        if pid == teacher_pid:
            return deterministic_ppo_action(teacher, simulation, pid)[0]
        return opponent_map[pid].choose_action(simulation)

    rows = []
    seen_states = set()
    max_decisions = env.max_rounds * NUM_PLAYERS * 30
    for step in range(max_decisions):
        if env.done:
            break
        pid = env.whose_turn()
        legal = env.get_allowed_actions(pid)
        if pid != teacher_pid:
            action = opponent_map[pid].choose_action(env)
            env.step(action if action in legal else fallback_action(legal))
            continue

        teacher_action, probabilities = deterministic_ppo_action(teacher, env, pid)
        if len(legal) > 1:
            prompt_hash = sha256_text(canonical_prompt(env, pid))
            if prompt_hash not in seen_states:
                seen_states.add(prompt_hash)
                candidates = shortlist_actions(
                    legal, teacher_action, probabilities, candidate_limit
                )
                rollout_seeds = [
                    seed * 1_000_003 + step * rollouts_per_action + index
                    for index in range(rollouts_per_action)
                ]
                scores = score_candidates(
                    env,
                    candidates,
                    pid,
                    rollout_policy,
                    rollout_seeds,
                    rollout_horizon,
                )
                relabeled_action, margin = best_relabel(scores, min_margin)
                if relabeled_action is not None:
                    row = make_dataset_row(
                        env=env,
                        actor_pid=pid,
                        game_id=game_id,
                        seed=seed,
                        step=step,
                        teacher_action=teacher_action,
                        relabeled_action=relabeled_action,
                        rollout_scores=scores,
                        outcome="pending",
                        teacher_checkpoint_hash=teacher_checkpoint_hash,
                    )
                    row["rollout_margin"] = margin
                    rows.append(row)
        env.step(teacher_action)

    winner = env.winner()
    outcome = "win" if winner == teacher_pid else "loss"
    for row in rows:
        row["outcome"] = outcome
        row["winner"] = winner
    return rows, {
        "game_id": game_id,
        "seed": seed,
        "teacher_seat": teacher_pid,
        "winner": winner,
        "outcome": outcome,
        "finished": env.done,
        "retained_rows": len(rows),
    }


def validate_dataset_row(row: Mapping) -> None:
    required = {
        "schema", "ruleset", "game_id", "seed", "step", "actor_pid",
        "seat_order", "phase",
        "action_family", "prompt", "completion", "teacher_action",
        "relabeled_action", "legal_actions", "rollout_scores", "outcome",
        "teacher_checkpoint_hash", "state_hash",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Dataset row is missing fields: {sorted(missing)}")
    if row["schema"] != SCHEMA_VERSION or row["ruleset"] != RULESET_VERSION:
        raise ValueError("Dataset schema or ruleset mismatch")
    if row["state_hash"] != sha256_text(row["prompt"]):
        raise ValueError("Dataset state hash mismatch")
    if row["relabeled_action"] not in row["legal_actions"]:
        raise ValueError("Relabeled action is illegal")
    parsed = json.loads(row["completion"])
    if not isinstance(parsed, dict) or parsed.get("action") is None:
        raise ValueError("Completion is not a canonical action object")
    validation_env = MonopolyEnv(agent_ids=[row["actor_pid"]], max_rounds=1)
    validation_env.turn_order = list(row["seat_order"])
    completion_action = object_to_action(parsed, validation_env, row["actor_pid"])
    if completion_action != row["relabeled_action"]:
        raise ValueError("Completion does not encode the relabeled action")
    candidate_ids = {int(action) for action in row["rollout_scores"]}
    if row["relabeled_action"] not in candidate_ids:
        raise ValueError("Relabeled action has no preserved rollout scores")


def _balanced_rows(rows: Iterable[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["phase"], row["action_family"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (row["state_hash"], row["step"]))
    result = []
    while buckets:
        for key in sorted(tuple(buckets)):
            result.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return result


def split_by_game(
    rows: Sequence[dict],
    sizes: Mapping[str, int] | None = None,
    seed: int = 3407,
) -> dict[str, list[dict]]:
    """Deduplicate states and fill exact splits without leaking a game across splits."""
    sizes = dict(sizes or {"train": 2048, "validation": 256, "test": 256})
    unique = {row["state_hash"]: dict(row) for row in rows}
    games: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        games[row["game_id"]].append(row)
    game_ids = sorted(games)
    random.Random(seed).shuffle(game_ids)
    result = {name: [] for name in sizes}
    cursor = 0
    for name, size in sizes.items():
        while len(result[name]) < size and cursor < len(game_ids):
            game_id = game_ids[cursor]
            cursor += 1
            available = _balanced_rows(games[game_id])
            result[name].extend(available[: size - len(result[name])])
        if len(result[name]) != size:
            raise ValueError(f"Insufficient unique game records for {name}: {len(result[name])}/{size}")
    return result


def validate_splits(splits: Mapping[str, Sequence[Mapping]]) -> None:
    games: dict[str, str] = {}
    states: set[str] = set()
    for split, rows in splits.items():
        for row in rows:
            validate_dataset_row(row)
            previous = games.setdefault(row["game_id"], split)
            if previous != split:
                raise ValueError(f"Game {row['game_id']} leaks across splits")
            if row["state_hash"] in states:
                raise ValueError(f"State {row['state_hash']} is duplicated")
            states.add(row["state_hash"])


def tokenize_rows(rows: Sequence[dict], tokenizer, max_length: int = 512) -> list[dict]:
    """Create response-only labels without truncation and enforce the 0.5% overflow gate."""
    tokenized = []
    overlong = 0
    for row in rows:
        prompt_messages = [{"role": "user", "content": row["prompt"]}]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": row["completion"]}
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenizer chat template does not preserve the prompt prefix")
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        if len(input_ids) > max_length:
            overlong += 1
        item = dict(row)
        item.update(
            input_ids=list(input_ids),
            attention_mask=[1] * len(input_ids),
            labels=labels,
            token_count=len(input_ids),
        )
        tokenized.append(item)
    if tokenized and overlong / len(tokenized) > 0.005:
        raise ValueError(
            f"Token overflow gate failed: {overlong}/{len(tokenized)} rows exceed {max_length}"
        )
    return tokenized


def scripted_opponents(seed: int, teacher_pid: int):
    classes = random.Random(seed).sample(FP_AGENT_CLASSES, NUM_PLAYERS - 1)
    seats = [pid for pid in range(NUM_PLAYERS) if pid != teacher_pid]
    return [agent_class(pid) for agent_class, pid in zip(classes, seats)]


def train_teacher_block(
    agent,
    games: int,
    *,
    seed: int,
    checkpoint_path: str | Path,
    checkpoint_every: int = 100,
    watchdog=None,
) -> list[dict]:
    """Train PPO against randomized three-of-six scripted groups."""
    history = []
    start = int(getattr(agent, "games_trained", 0))
    env = MonopolyEnv(agent_ids=[agent.player_id], max_rounds=200)
    for offset in range(1, games + 1):
        game = start + offset
        episode_seed = seed + game - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if watchdog is not None:
            watchdog.check()
        opponents = scripted_opponents(episode_seed, agent.player_id)
        result = run_episode(env, agent, opponents, agent.player_id, is_ppo=True)
        agent.games_trained = game
        history.append({"game": game, **result})
        if game % checkpoint_every == 0:
            agent.save(str(checkpoint_path))
    agent.save(str(checkpoint_path))
    return history


def _play_policy_game(
    env: MonopolyEnv,
    teacher_pid: int,
    teacher_policy: Callable[[MonopolyEnv, int], int],
    opponents,
) -> int:
    env.reset()
    opponent_map = {agent.player_id: agent for agent in opponents}
    max_decisions = env.max_rounds * NUM_PLAYERS * 30
    for _ in range(max_decisions):
        if env.done:
            break
        pid = env.whose_turn()
        legal = env.get_allowed_actions(pid)
        action = (
            teacher_policy(env, pid)
            if pid == teacher_pid
            else opponent_map[pid].choose_action(env)
        )
        env.step(action if action in legal else fallback_action(legal))
    return env.winner()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        raise ValueError("Wilson interval requires at least one game")
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return center - radius, center + radius


def evaluate_teacher(agent, games: int = 600, seed: int = 90_000) -> dict:
    wins = 0
    records = []
    for index in range(games):
        game_seed = seed + index
        teacher_pid = index % NUM_PLAYERS
        random.seed(game_seed)
        np.random.seed(game_seed)
        env = MonopolyEnv(agent_ids=[teacher_pid], max_rounds=200)
        opponents = scripted_opponents(game_seed, teacher_pid)
        winner = _play_policy_game(
            env,
            teacher_pid,
            lambda state, pid: deterministic_ppo_action(agent, state, pid)[0],
            opponents,
        )
        wins += winner == teacher_pid
        records.append({"seed": game_seed, "seat": teacher_pid, "winner": winner})
    low, high = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "wilson_95": [low, high],
        "passed": wins / games >= 0.35 and low > 0.25,
        "records": records,
    }


def play_model_game(
    env: MonopolyEnv,
    model_pid: int,
    generate: Callable[[str], str],
    opponents,
) -> dict:
    """Run Gemma as the standalone policy; only single-action states are automatic."""
    env.reset()
    opponent_map = {agent.player_id: agent for agent in opponents}
    decisions = []
    max_decisions = env.max_rounds * NUM_PLAYERS * 30
    for step in range(max_decisions):
        if env.done:
            break
        pid = env.whose_turn()
        legal = env.get_allowed_actions(pid)
        if pid != model_pid:
            action = opponent_map[pid].choose_action(env)
            env.step(action if action in legal else fallback_action(legal))
            continue
        if len(legal) == 1:
            env.step(legal[0])
            continue
        prompt = canonical_prompt(env, model_pid)
        started = time.perf_counter()
        raw = generate(prompt)
        latency = time.perf_counter() - started
        action, error = parse_or_fallback(raw, env, model_pid)
        decisions.append(
            {
                "step": step,
                "prompt": prompt,
                "raw_output": raw,
                "parsed_action": action,
                "fallback": error,
                "latency_s": latency,
            }
        )
        env.step(action)
    return {
        "winner": env.winner(),
        "model_won": env.winner() == model_pid,
        "finished": env.done,
        "standings": sorted(
            (
                {"seat": pid, "net_worth": player.net_worth(), "bankrupt": player.bankrupt}
                for pid, player in enumerate(env.players)
            ),
            key=lambda row: row["net_worth"],
            reverse=True,
        ),
        "decisions": decisions,
    }


__all__ = [
    "SCHEMA_VERSION", "SYSTEM_PROMPT", "DecisionFormatError", "action_family",
    "action_to_json", "action_to_object", "best_relabel", "canonical_prompt",
    "canonical_state", "collect_relabelled_game", "deterministic_ppo_action", "evaluate_teacher",
    "fallback_action", "file_sha256", "grouped_legal_actions", "make_dataset_row",
    "object_to_action", "parse_action_json", "parse_or_fallback", "play_model_game",
    "rollout_value", "score_candidates", "scripted_opponents", "seat_names",
    "seat_order", "serialize_decision", "sha256_text", "shortlist_actions",
    "split_by_game", "tokenize_rows", "train_teacher_block", "validate_dataset_row",
    "validate_splits", "wilson_interval",
]
