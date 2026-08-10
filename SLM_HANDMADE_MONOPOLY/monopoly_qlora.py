"""Shared contracts for Gemma Monopoly ASU-teacher data and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ASU_FROZEN_TEACHER import (  # noqa: E402
    ASU_VALUE_V1,
    FROZEN_SPEC_HASH,
    ASUValueV1,
)

from monopoly_game_engine.actions import (  # noqa: E402
    ACTION_SPACE_SIZE,
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC  # noqa: E402
from monopoly_game_engine.constants import (  # noqa: E402
    NUM_PLAYERS,
    RULESET_VERSION,
    TRADE_CASH_LEVELS,
)
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402


SCHEMA_VERSION = "monopoly-decision-v2"
ASU_SOURCE_ROOT = PROJECT_ROOT / "ASU_FROZEN_TEACHER"
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
    "Monopoly: SELF=0 and OPPn=n. T=phase/round/active/rolled/dice; "
    "PL seats=position/cash/worth/jail/turns/card/bankrupt (tail 0); "
    "D=alias/owner/mortgage/houses:squares (@ joins). B/PR/BT/ST/X=legal "
    "domains; trade max=75/100/125. Return legal JSON only."
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


def asu_teacher_hash(config: Mapping | None = None) -> str:
    """Hash the frozen ASU policy implementation, spec, and collection config."""
    return sha256_text(canonical_json({
        "policy": ASU_VALUE_V1,
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "core_sha256": file_sha256(ASU_SOURCE_ROOT / "core.py"),
        "spec_sha256": file_sha256(ASU_SOURCE_ROOT / "spec.py"),
        "types_sha256": file_sha256(ASU_SOURCE_ROOT / "types.py"),
        "init_sha256": file_sha256(ASU_SOURCE_ROOT / "__init__.py"),
        "ruleset": RULESET_VERSION,
        "config": dict(config or {}),
    }))


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


def compact_state_payload(payload: Mapping) -> dict:
    """Remove repeated seat and legal-domain data without dropping choices."""
    if "p" in payload:
        return dict(payload)

    seat_ids = {
        player[0]: index for index, player in enumerate(payload["players"])
    }
    deed_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for square, owner, mortgaged, houses in payload["deeds"]:
        deed_groups[(seat_ids[owner], mortgaged, houses)].append(square)

    legal = payload["legal"]
    compact_legal: dict[str, object] = {}
    if legal.get("binary"):
        compact_legal["binary"] = legal["binary"]
    if legal.get("auction_bid"):
        compact_legal["auction_bid"] = legal["auction_bid"]

    property_domains: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for action in PROPERTY_ACTIONS:
        if legal.get(action):
            property_domains[tuple(legal[action])].append(action)
    if property_domains:
        compact_legal["property"] = [
            [actions, list(squares)]
            for squares, actions in sorted(
                property_domains.items(), key=lambda item: item[1]
            )
        ]

    for action in ("buy_trade", "sell_trade"):
        entries = []
        for target, price_domains in sorted(legal.get(action, {}).items()):
            for prices, squares in sorted(
                price_domains.items(),
                key=lambda item: tuple(map(int, item[0].split(","))),
            ):
                entries.append([target, list(map(int, prices.split(","))), squares])
        if entries:
            compact_legal[action] = entries

    exchange = legal.get("exchange_trade")
    if exchange:
        offer_domains = {tuple(domain["offer"]) for domain in exchange.values()}
        if len(offer_domains) != 1:
            raise ValueError("Exchange targets have inconsistent offer domains")
        compact_legal["exchange_trade"] = [
            list(next(iter(offer_domains))),
            [[target, domain["request"]] for target, domain in sorted(exchange.items())],
        ]

    turn = [
        payload["phase"],
        payload["round"],
        payload["active"],
        payload["rolled"],
        *payload["dice"],
    ]
    compact = {
        "turn": turn,
        "bank": payload["supply"],
        "p": [player[1:] for player in payload["players"]],
        "d": [
            [owner, mortgaged, houses, squares]
            for (owner, mortgaged, houses), squares in sorted(deed_groups.items())
        ],
        "legal": compact_legal,
    }
    if payload["debt"][0]:
        compact["debt"] = payload["debt"]
    if payload["trade"] is not None:
        trade = payload["trade"]
        compact["trade"] = [
            trade["from"], trade["offer"], trade["request"],
            trade["cash_offer"], trade["cash_request"],
        ]
    if payload["auction"] is not None:
        auction = payload["auction"]
        compact["auction"] = [
            auction["square"], auction["high_bid"],
            auction["leader"], auction["bidders"],
        ]
    return compact


def compact_payload_text(payload: Mapping) -> str:
    """Encode the compact public state and complete legal domains as a small DSL."""
    seats = {"SELF": 0, "OPP1": 1, "OPP2": 2, "OPP3": 3, None: "-"}

    def joined(values, separator="/"):
        return separator.join(str(seats.get(value, value)) for value in values)

    turn = list(payload["turn"])
    turn[2] = seats[turn[2]]
    if not turn[3]:
        turn = turn[:4]
    while len(turn) > 2 and turn[-1] == 0:
        turn.pop()
    lines = [f"T={joined(turn)}", f"K={joined(payload['bank'])}"]

    players = []
    for values in payload["p"]:
        values = list(values)
        while len(values) > 3 and values[-1] == 0:
            values.pop()
        players.append(joined(values))
    lines.append(f"PL={';'.join(players)}")
    deed_aliases = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(payload["d"]) > len(deed_aliases):
        raise ValueError("Too many deed domains for the compact serializer")
    deed_domains = [set(group[3]) for group in payload["d"]]

    def domain(squares):
        wanted = set(squares)
        references = [
            deed_aliases[index]
            for index, values in enumerate(deed_domains)
            if values and values <= wanted
        ]
        covered = set().union(
            *(deed_domains[deed_aliases.index(alias)] for alias in references)
        ) if references else set()
        return (
            "@" + "".join(references)
            if covered == wanted
            else joined(squares, ",")
        )

    def max_price(prices):
        if tuple(prices) != PRICE_PCTS[: len(prices)]:
            raise ValueError("Trade price domain is not a standard prefix")
        return prices[-1]

    lines.append(
        "D=" + ";".join(
            f"{deed_aliases[index]}={joined(group[:3])}:{joined(group[3], ',')}"
            for index, group in enumerate(payload["d"])
        )
    )
    if "debt" in payload:
        lines.append(f"DE={joined(payload['debt'])}")
    if "trade" in payload:
        lines.append(f"TR={joined(payload['trade'])}")
    if "auction" in payload:
        auction = list(payload["auction"])
        auction[2] = seats[auction[2]]
        auction[3] = joined(auction[3], ",")
        lines.append(f"AU={joined(auction)}")

    legal = payload["legal"]
    if "binary" in legal:
        lines.append(f"B={joined(legal['binary'], ',')}")
    if "auction_bid" in legal:
        lines.append(f"AB={joined(legal['auction_bid'], ',')}")
    if "property" in legal:
        lines.append(
            "PR=" + ";".join(
                f"{joined(actions, '+')}:{domain(squares)}"
                for actions, squares in legal["property"]
            )
        )
    for action, label in (("buy_trade", "BT"), ("sell_trade", "ST")):
        if action in legal:
            target_domains: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for target, prices, squares in legal[action]:
                target_domains[target].append((max_price(prices), domain(squares)))
            shared_domains: dict[tuple[tuple[int, str], ...], list[str]] = defaultdict(list)
            for target, entries in target_domains.items():
                shared_domains[tuple(entries)].append(target)
            lines.append(
                label + "=" + ";".join(
                    "+".join(str(seats[target]) for target in targets)
                    + "/"
                    + "|".join(f"{price}:{squares}" for price, squares in entries)
                    for entries, targets in shared_domains.items()
                )
            )
    if "exchange_trade" in legal:
        offers, requests = legal["exchange_trade"]
        lines.append(
            f"X={domain(offers)}:" + ";".join(
                f"{seats[target]}={domain(squares)}"
                for target, squares in requests
            )
        )
    return "\n".join(lines)


def serialize_decision(env: MonopolyEnv, actor_pid: int) -> str:
    payload = canonical_state(env, actor_pid)
    payload["legal"] = grouped_legal_actions(env, actor_pid)
    return compact_payload_text(compact_state_payload(payload))


def canonical_prompt(env: MonopolyEnv, actor_pid: int) -> str:
    return f"{SYSTEM_PROMPT}\n{serialize_decision(env, actor_pid)}"


def compact_dataset_prompt(prompt: str) -> str:
    """Upgrade a saved verbose prompt to the current lossless compact form."""
    _, serialized = prompt.split("\n", 1)
    if not serialized.startswith("{"):
        return prompt
    payload = compact_state_payload(json.loads(serialized))
    return f"{SYSTEM_PROMPT}\n{compact_payload_text(payload)}"


def shortlist_actions(
    legal_actions: Sequence[int],
    teacher_action: int,
    scores: Mapping[int, float],
    eligible_actions: Sequence[int],
    mandatory_actions: Sequence[int],
    limit: int = 16,
) -> list[int]:
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    legal = list(dict.fromkeys(int(action) for action in legal_actions))
    if teacher_action not in legal:
        raise ValueError("Teacher action must be legal")
    pool = [int(action) for action in eligible_actions if action in legal]
    if teacher_action not in pool:
        pool.insert(0, teacher_action)
    selected = [teacher_action]

    def add_family_representatives(actions: Iterable[int]) -> None:
        by_family: dict[str, list[int]] = defaultdict(list)
        for action in actions:
            if action in pool:
                by_family[action_family(action)].append(action)
        representatives = (
            max(actions, key=lambda action: (scores[action], -action))
            for actions in by_family.values()
        )
        for action in sorted(
            representatives, key=lambda action: (-scores[action], action)
        ):
            if len(selected) >= limit:
                return
            if action not in selected:
                selected.append(action)

    add_family_representatives(dict.fromkeys(int(item) for item in mandatory_actions))
    add_family_representatives(pool)
    remaining = sorted(
        (action for action in pool if action not in selected),
        key=lambda action: (-scores[action], action),
    )
    return (selected + remaining)[:limit]


def asu_teacher_decision(
    env: MonopolyEnv,
    actor_pid: int,
    candidate_limit: int = 16,
) -> tuple[int, dict[int, dict]]:
    """Return ASU's legal action and compact top candidate records."""
    decision = ASUValueV1(actor_pid).decide(env)
    legal = env.get_allowed_actions(actor_pid)
    by_action = {candidate.action: candidate for candidate in decision.candidates}
    if set(by_action) != set(legal) or decision.selected_action not in legal:
        raise RuntimeError("ASU decision does not exactly cover the legal action set")
    scores = {action: float(candidate.score) for action, candidate in by_action.items()}
    selected = shortlist_actions(
        legal,
        decision.selected_action,
        scores,
        [candidate.action for candidate in decision.candidates if candidate.eligible],
        [candidate.action for candidate in decision.candidates if candidate.mandatory],
        candidate_limit,
    )
    records = {}
    for action in selected:
        record = asdict(by_action[action])
        record["score"] = scores[action]
        records[action] = record
    return int(decision.selected_action), records


def exploratory_behavior_action(
    teacher_action: int,
    teacher_candidates: Mapping[int, Mapping],
    *,
    seed: int,
    step: int,
    every: int = 9,
    top_k: int = 3,
) -> tuple[int, bool]:
    """Occasionally execute a safe ASU alternative while retaining ASU's label."""
    if every < 1 or top_k < 1:
        raise ValueError("Exploration cadence and top-k must be positive")
    candidates = {int(action): value for action, value in teacher_candidates.items()}
    selected = candidates[teacher_action]
    if selected["forced"] or selected["mandatory"]:
        return teacher_action, False
    alternatives = sorted(
        (
            action
            for action, value in candidates.items()
            if action != teacher_action
            and value["eligible"]
            and not value["forced"]
            and not value["mandatory"]
        ),
        key=lambda action: (-float(candidates[action]["score"]), action),
    )[:top_k]
    rng = random.Random(seed * 1_000_003 + step)
    if not alternatives or rng.randrange(every):
        return teacher_action, False
    return rng.choice(alternatives), True


def make_dataset_row(
    *,
    env: MonopolyEnv,
    actor_pid: int,
    game_id: str,
    seed: int,
    step: int,
    teacher_policy: str,
    teacher_candidates: Mapping[int, Mapping],
    teacher_action: int,
    behavior_action: int,
    exploratory: bool,
    relabeled_action: int,
    outcome: str,
    teacher_bundle_hash: str,
) -> dict:
    legal = env.get_allowed_actions(actor_pid)
    candidates = {int(action): dict(value) for action, value in teacher_candidates.items()}
    if type(teacher_policy) is not str or not teacher_policy:
        raise ValueError("Teacher policy must be identified")
    if any(action not in legal for action in candidates):
        raise ValueError("Teacher candidates must be legal")
    if (
        teacher_action not in legal
        or behavior_action not in legal
        or relabeled_action not in legal
    ):
        raise ValueError("Dataset actions must be legal in the recorded state")
    if any(action not in candidates for action in (teacher_action, behavior_action, relabeled_action)):
        raise ValueError("Dataset labels must preserve their teacher candidate scores")
    if bool(exploratory) != (behavior_action != teacher_action):
        raise ValueError("Exploration flag and behavior action disagree")
    if teacher_policy == ASU_VALUE_V1 and relabeled_action != teacher_action:
        raise ValueError("ASU labels must equal the ASU teacher action")
    if len(teacher_bundle_hash) != 64 or any(
        character not in "0123456789abcdef" for character in teacher_bundle_hash
    ):
        raise ValueError("Teacher hash must be lowercase SHA-256")
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
        "teacher_policy": teacher_policy,
        "teacher_action": int(teacher_action),
        "behavior_action": int(behavior_action),
        "exploratory": bool(exploratory),
        "relabeled_action": int(relabeled_action),
        "legal_actions": sorted(int(action) for action in legal),
        "teacher_candidates": {
            str(action): value for action, value in sorted(candidates.items())
        },
        "candidate_scores": {
            str(action): float(value["score"])
            for action, value in sorted(candidates.items())
        },
        "outcome": outcome,
        "teacher_bundle_hash": teacher_bundle_hash,
        "state_hash": sha256_text(prompt),
    }


def collect_teacher_game(
    *,
    env: MonopolyEnv,
    teacher_pid: int,
    opponents,
    game_id: str,
    seed: int,
    teacher_bundle_hash: str,
    candidate_limit: int = 16,
    exploration_every: int = 9,
    exploration_top_k: int = 3,
    watchdog=None,
) -> tuple[list[dict], dict]:
    """Collect non-forced ASU decisions against randomized A/B/C opponents."""
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    opponent_map = {agent.player_id: agent for agent in opponents}

    rows = []
    seen_states = set()
    perturbations = 0
    max_decisions = env.max_rounds * NUM_PLAYERS * 30
    for step in range(max_decisions):
        if watchdog is not None:
            watchdog.check()
        if env.done:
            break
        pid = env.whose_turn()
        legal = env.get_allowed_actions(pid)
        if pid != teacher_pid:
            action = opponent_map[pid].choose_action(env)
            env.step(action if action in legal else fallback_action(legal))
            continue
        if len(legal) == 1:
            env.step(legal[0])
            continue

        teacher_action, teacher_candidates = asu_teacher_decision(
            env, pid, candidate_limit
        )
        behavior_action, exploratory = exploratory_behavior_action(
            teacher_action,
            teacher_candidates,
            seed=seed,
            step=step,
            every=exploration_every,
            top_k=exploration_top_k,
        )
        perturbations += exploratory
        prompt_hash = sha256_text(canonical_prompt(env, pid))
        if prompt_hash not in seen_states:
            seen_states.add(prompt_hash)
            rows.append(make_dataset_row(
                env=env,
                actor_pid=pid,
                game_id=game_id,
                seed=seed,
                step=step,
                teacher_policy=ASU_VALUE_V1,
                teacher_candidates=teacher_candidates,
                teacher_action=teacher_action,
                behavior_action=behavior_action,
                exploratory=exploratory,
                relabeled_action=teacher_action,
                outcome="pending",
                teacher_bundle_hash=teacher_bundle_hash,
            ))
        env.step(behavior_action)

    winner = env.winner()
    outcome = "win" if winner == teacher_pid else "loss"
    for row in rows:
        row["outcome"] = outcome
        row["winner"] = winner
    return rows, {
        "game_id": game_id,
        "seed": seed,
        "teacher_seat": teacher_pid,
        "teacher_policy": ASU_VALUE_V1,
        "winner": winner,
        "outcome": outcome,
        "finished": env.done,
        "retained_rows": len(rows),
        "exploratory_actions": perturbations,
    }


def validate_dataset_row(row: Mapping) -> None:
    required = {
        "schema", "ruleset", "game_id", "seed", "step", "actor_pid",
        "seat_order", "phase",
        "action_family", "prompt", "completion", "teacher_policy",
        "teacher_action", "behavior_action", "exploratory",
        "relabeled_action", "legal_actions",
        "teacher_candidates", "candidate_scores", "outcome",
        "teacher_bundle_hash", "state_hash",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Dataset row is missing fields: {sorted(missing)}")
    if row["schema"] != SCHEMA_VERSION or row["ruleset"] != RULESET_VERSION:
        raise ValueError("Dataset schema or ruleset mismatch")
    if type(row["teacher_policy"]) is not str or not row["teacher_policy"]:
        raise ValueError("Teacher policy must be identified")
    if row["state_hash"] != sha256_text(row["prompt"]):
        raise ValueError("Dataset state hash mismatch")
    if row["relabeled_action"] not in row["legal_actions"]:
        raise ValueError("Relabeled action is illegal")
    if row["teacher_action"] not in row["legal_actions"]:
        raise ValueError("Teacher action is illegal")
    if row["behavior_action"] not in row["legal_actions"]:
        raise ValueError("Behavior action is illegal")
    if bool(row["exploratory"]) != (
        row["behavior_action"] != row["teacher_action"]
    ):
        raise ValueError("Exploration flag and behavior action disagree")
    if (
        row["teacher_policy"] == ASU_VALUE_V1
        and row["relabeled_action"] != row["teacher_action"]
    ):
        raise ValueError("ASU labels must equal the ASU teacher action")
    parsed = json.loads(row["completion"])
    if not isinstance(parsed, dict) or parsed.get("action") is None:
        raise ValueError("Completion is not a canonical action object")
    validation_env = MonopolyEnv(agent_ids=[row["actor_pid"]], max_rounds=1)
    validation_env.turn_order = list(row["seat_order"])
    completion_action = object_to_action(parsed, validation_env, row["actor_pid"])
    if completion_action != row["relabeled_action"]:
        raise ValueError("Completion does not encode the relabeled action")
    candidate_ids = {int(action) for action in row["candidate_scores"]}
    if row["relabeled_action"] not in candidate_ids:
        raise ValueError("Relabeled action has no preserved candidate score")
    if row["teacher_action"] not in candidate_ids:
        raise ValueError("Teacher action has no preserved candidate score")
    if row["behavior_action"] not in candidate_ids:
        raise ValueError("Behavior action has no preserved candidate score")
    if set(row["teacher_candidates"]) != set(row["candidate_scores"]):
        raise ValueError("Teacher candidate records and scores disagree")
    for action, score in row["candidate_scores"].items():
        if float(row["teacher_candidates"][action]["score"]) != float(score):
            raise ValueError("Teacher candidate record has a mismatched score")
    if len(row["teacher_bundle_hash"]) != 64 or any(
        character not in "0123456789abcdef"
        for character in row["teacher_bundle_hash"]
    ):
        raise ValueError("Teacher hash must be lowercase SHA-256")


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
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        context_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        input_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        prefix_length = 0
        for prompt_token, input_token in zip(prompt_ids, input_ids):
            if prompt_token != input_token:
                break
            prefix_length += 1
        if (
            input_ids[: len(context_ids)] != context_ids
            or prefix_length < len(context_ids)
        ):
            raise ValueError("Tokenizer chat template has no usable response boundary")
        labels = [-100] * prefix_length + input_ids[prefix_length:]
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
    classes = random.Random(seed).sample(
        (FPAgentA, FPAgentB, FPAgentC), NUM_PLAYERS - 1
    )
    seats = [pid for pid in range(NUM_PLAYERS) if pid != teacher_pid]
    return [agent_class(pid) for agent_class, pid in zip(classes, seats)]


def play_policy_game(
    env: MonopolyEnv,
    teacher_pid: int,
    teacher_policy: Callable[[MonopolyEnv, int], int],
    opponents,
) -> dict:
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
    return {"winner": env.winner(), "finished": env.done}


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
    "action_to_json", "action_to_object", "asu_teacher_decision",
    "asu_teacher_hash", "canonical_prompt", "canonical_state",
    "collect_teacher_game", "exploratory_behavior_action",
    "fallback_action", "file_sha256", "grouped_legal_actions", "make_dataset_row",
    "object_to_action", "parse_action_json", "parse_or_fallback", "play_model_game",
    "play_policy_game", "scripted_opponents", "seat_names",
    "seat_order", "serialize_decision", "sha256_text", "shortlist_actions",
    "split_by_game", "tokenize_rows", "validate_dataset_row", "validate_splits",
]
