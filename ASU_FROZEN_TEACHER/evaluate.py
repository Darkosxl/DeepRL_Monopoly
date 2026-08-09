"""Seat-balanced evaluation CLI for the frozen ASU teachers and baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    ASURolloutV1,
    ASUValueV1,
    RULESET_VERSION,
    _PrivateGame,
    preserve_global_rng,
)
from .spec import FROZEN_SPEC_HASH

from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType  # noqa: E402
from monopoly_game_engine.agent_ddqn import DDQNAgent  # noqa: E402
from monopoly_game_engine.agent_ppo import (  # noqa: E402
    PPOAgent,
    fixed_accept_trade_decision,
    fixed_buy_decision,
)
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402


SUPPORTED_SCRIPTED = tuple(f"fixed-{letter}" for letter in "abcdef")
SUPPORTED_ASU = ("asu-value-v1", "asu-rollout-v1")
CHECKPOINT_KINDS = ("ppo", "ddqn", "cfr")
DEFAULT_MAX_DECISIONS = 20_000
RESULTS_DISCLAIMER = (
    "These are ppo-plus-v2 simulator results for ASU-inspired reconstructions; "
    "they do not reproduce or estimate either cited paper's reported win rates."
)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    kind: str
    policy_id: str
    checkpoint: Path | None = None


def parse_agent_spec(value: str) -> AgentSpec:
    """Parse a scripted/ASU ID or ``ppo|ddqn|cfr:/path`` checkpoint spec."""

    normalized = value.strip()
    if normalized in SUPPORTED_ASU or normalized in SUPPORTED_SCRIPTED:
        return AgentSpec(normalized, normalized)
    for separator in (":", "="):
        kind, found, path_text = normalized.partition(separator)
        if found and kind in CHECKPOINT_KINDS:
            if not path_text:
                raise ValueError(f"{kind} requires an explicit checkpoint path")
            checkpoint = Path(path_text).expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"Missing {kind.upper()} checkpoint: {checkpoint}")
            return AgentSpec(kind, f"{kind}:{checkpoint}", checkpoint)
    supported = ", ".join((*SUPPORTED_ASU, *SUPPORTED_SCRIPTED))
    raise ValueError(
        f"Unknown agent spec {value!r}; use {supported}, or ppo:/path, "
        "ddqn:/path, cfr:/path"
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(wins: int, games: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval."""

    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


class _NeuralAdapter:
    def __init__(self, agent, player_id: int, kind: str):
        self.agent = agent
        self.player_id = player_id
        self.kind = kind

    def _hybrid_choice(self, env, allowed: list[int]) -> int | None:
        if not self.agent.hybrid:
            return None
        if int(ActionType.BUY_PROPERTY) in allowed and fixed_buy_decision(
            env, self.player_id
        ):
            return int(ActionType.BUY_PROPERTY)
        if (
            int(ActionType.ACCEPT_TRADE) in allowed
            and env._incoming_trade(self.player_id) is not None
        ):
            return (
                int(ActionType.ACCEPT_TRADE)
                if fixed_accept_trade_decision(env, self.player_id)
                else int(ActionType.DECLINE_TRADE)
            )
        return None

    def choose_action(self, env) -> int:
        import torch

        allowed = list(env.get_allowed_actions(self.player_id))
        hybrid = self._hybrid_choice(env, allowed)
        if hybrid is not None:
            return hybrid
        if self.agent.hybrid:
            fixed = {int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE)}
            allowed = [action for action in allowed if action not in fixed]
        if not allowed:
            return int(ActionType.DO_NOTHING)

        state = env._get_state(self.player_id)
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.agent.device
        ).unsqueeze(0)
        with torch.inference_mode():
            if self.kind == "ppo":
                mask = torch.zeros(
                    (1, ACTION_SPACE_SIZE),
                    dtype=torch.bool,
                    device=self.agent.device,
                )
                mask[0, allowed] = True
                scores = self.agent.actor(state_tensor, mask).squeeze(0)
            else:
                scores = self.agent.online_net(state_tensor).squeeze(0)
                illegal = torch.ones(
                    ACTION_SPACE_SIZE,
                    dtype=torch.bool,
                    device=self.agent.device,
                )
                illegal[allowed] = False
                scores = scores.masked_fill(illegal, float("-inf"))
        return int(scores.argmax().item())


class _CFRAdapter:
    def __init__(self, model, player_id: int):
        self.model = model
        self.player_id = player_id

    def choose_action(self, env) -> int:
        from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr import (  # noqa: E501
            SharedGame,
        )

        game = SharedGame(env=env, random_state=random.getstate())
        actions = game.legal_actions()
        node = self.model.tables[self.player_id].get(
            game.information_key(self.player_id)
        )
        if node is None:
            return min(actions)
        probabilities = node.average_policy()
        maximum = float(probabilities.max())
        return min(
            action
            for action, probability in zip(actions, probabilities)
            if math.isclose(float(probability), maximum, rel_tol=0.0, abs_tol=1e-12)
        )


class _ScriptedAdapter:
    """Match the compatibility fallback used by the repository's game drivers."""

    def __init__(self, agent, player_id: int):
        self.agent = agent
        self.player_id = player_id
        self.fallbacks = 0

    def choose_action(self, env) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        action = self.agent.choose_action(env)
        if action in allowed:
            return action
        self.fallbacks += 1
        if int(ActionType.END_TURN) in allowed:
            return int(ActionType.END_TURN)
        return allowed[0]


class AgentFactory:
    """Load each checkpoint once and bind read-only policies to physical seats."""

    def __init__(self):
        self._loaded: dict[tuple[str, Path], Any] = {}

    @staticmethod
    def _torch_metadata(path: Path, kind: str) -> dict[str, Any]:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Unable to load {kind.upper()} checkpoint {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Incompatible {kind.upper()} checkpoint {path}: expected a mapping"
            )
        expected = {
            "format_version": 3,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
        }
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Incompatible {kind.upper()} checkpoint metadata at {path}: "
                f"{actual}; expected {expected}"
            )
        for key in ("player_id", "hybrid", "hidden_dim"):
            if key not in payload:
                raise ValueError(
                    f"Incompatible {kind.upper()} checkpoint {path}: missing {key}"
                )
        return payload

    def _load(self, spec: AgentSpec):
        assert spec.checkpoint is not None
        key = (spec.kind, spec.checkpoint)
        if key in self._loaded:
            return self._loaded[key]
        if spec.kind in ("ppo", "ddqn"):
            metadata = self._torch_metadata(spec.checkpoint, spec.kind)
            constructor = PPOAgent if spec.kind == "ppo" else DDQNAgent
            agent = constructor(
                player_id=int(metadata["player_id"]),
                hybrid=bool(metadata["hybrid"]),
                hidden_dim=int(metadata["hidden_dim"]),
                device="cpu",
            )
            try:
                agent.load(str(spec.checkpoint))
            except Exception as exc:
                raise ValueError(
                    f"Incompatible {spec.kind.upper()} checkpoint {spec.checkpoint}: {exc}"
                ) from exc
            if spec.kind == "ppo":
                agent.actor.eval()
                agent.critic.eval()
            else:
                agent.online_net.eval()
                agent.target_net.eval()
            loaded = agent
        else:
            try:
                from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr import (  # noqa: E501
                    MonteCarloCFR,
                )

                loaded = MonteCarloCFR.load(spec.checkpoint)
            except Exception as exc:
                raise ValueError(
                    f"Incompatible CFR checkpoint {spec.checkpoint}: {exc}"
                ) from exc
        self._loaded[key] = loaded
        return loaded

    def build(self, spec: AgentSpec, player_id: int):
        if spec.kind == "asu-value-v1":
            return ASUValueV1(player_id)
        if spec.kind == "asu-rollout-v1":
            return ASURolloutV1(player_id)
        if spec.kind in SUPPORTED_SCRIPTED:
            agent = FP_AGENT_CLASSES[ord(spec.kind[-1]) - ord("a")](player_id)
            return _ScriptedAdapter(agent, player_id)
        model = self._load(spec)
        if spec.kind in ("ppo", "ddqn"):
            return _NeuralAdapter(model, player_id, spec.kind)
        return _CFRAdapter(model, player_id)


def _new_seeded_game(seed: int) -> _PrivateGame:
    outer = random.getstate()
    try:
        random.seed(seed)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        state_after_setup = random.getstate()
    finally:
        random.setstate(outer)
    game = _PrivateGame(env, seed)
    game.random_state = state_after_setup
    return game


def _run_game(
    specs: tuple[AgentSpec, ...],
    focus_seat: int,
    seed: int,
    max_decisions: int,
    factory: AgentFactory,
) -> dict[str, Any]:
    started = time.perf_counter()
    game = _new_seeded_game(seed)
    agents = [factory.build(spec, seat) for seat, spec in enumerate(specs)]
    decisions = 0
    while not game.env.done and decisions < max_decisions:
        actor = game.env.whose_turn()
        allowed = game.env.get_allowed_actions(actor)
        action = agents[actor].choose_action(game.env)
        if action not in allowed:
            raise RuntimeError(
                f"{specs[actor].policy_id} returned illegal action {action} "
                f"for seat {actor}; legal actions are {allowed}"
            )
        game.step(action)
        decisions += 1
    provisional_leader = game.env.winner()
    winner = provisional_leader if game.env.done else None
    scripted_fallbacks = [int(getattr(agent, "fallbacks", 0)) for agent in agents]
    return {
        "seed": seed,
        "focus_seat": focus_seat,
        "policies": [spec.policy_id for spec in specs],
        "winner": winner,
        "winner_policy": None if winner is None else specs[winner].policy_id,
        "provisional_leader": provisional_leader,
        "focus_won": None if winner is None else winner == focus_seat,
        "rounds": game.env.round,
        "decisions": decisions,
        "truncated": not game.env.done,
        "scripted_compatibility_fallbacks": scripted_fallbacks,
        "final_net_worth": [float(player.net_worth()) for player in game.env.players],
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_lineup(
    focus: AgentSpec | str,
    opponents: tuple[AgentSpec | str, AgentSpec | str, AgentSpec | str],
    seeds: tuple[int, ...] = (0,),
    max_decisions: int = DEFAULT_MAX_DECISIONS,
) -> dict[str, Any]:
    """Evaluate paired four-game seed blocks with the focus in every seat."""

    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    if not seeds:
        raise ValueError("at least one paired seed is required")
    if len(opponents) != NUM_PLAYERS - 1:
        raise ValueError(f"exactly {NUM_PLAYERS - 1} opponents are required")
    focus_spec = parse_agent_spec(focus) if isinstance(focus, str) else focus
    opponent_specs = tuple(
        parse_agent_spec(spec) if isinstance(spec, str) else spec for spec in opponents
    )
    factory = AgentFactory()
    games = []
    started = time.perf_counter()
    with preserve_global_rng():
        for seed in seeds:
            for focus_seat in range(NUM_PLAYERS):
                seats: list[AgentSpec | None] = [None] * NUM_PLAYERS
                seats[focus_seat] = focus_spec
                for seat, opponent in zip(
                    (seat for seat in range(NUM_PLAYERS) if seat != focus_seat),
                    opponent_specs,
                ):
                    seats[seat] = opponent
                games.append(
                    _run_game(
                        tuple(spec for spec in seats if spec is not None),
                        focus_seat,
                        int(seed),
                        max_decisions,
                        factory,
                    )
                )

    identifiers = {focus_spec.policy_id, *(spec.policy_id for spec in opponent_specs)}
    summaries = {}
    net_worth = {}
    for identifier in sorted(identifiers):
        appearances = [
            (game, seat)
            for game in games
            if not game["truncated"]
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        wins = sum(game["winner"] == seat for game, seat in appearances)
        interval = wilson_interval(wins, len(appearances))
        all_appearances = [
            (game, seat)
            for game in games
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        worth = [game["final_net_worth"][seat] for game, seat in all_appearances]
        rate = wins / len(appearances) if appearances else None
        summaries[identifier] = {
            "wins": wins,
            "games": len(appearances),
            "truncated_appearances": len(all_appearances) - len(appearances),
            "win_rate": rate,
            "win_rate_percent": None if rate is None else 100 * rate,
            "wilson_95": list(interval),
        }
        net_worth[identifier] = {
            "mean": sum(worth) / len(worth),
            "values": worth,
        }

    all_specs = (focus_spec, *opponent_specs)
    hashes = {
        spec.policy_id: checkpoint_sha256(spec.checkpoint)
        for spec in all_specs
        if spec.checkpoint is not None
    }
    return {
        "ruleset": RULESET_VERSION,
        "policy_ids": {
            "focus": focus_spec.policy_id,
            "opponents": [spec.policy_id for spec in opponent_specs],
        },
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "checkpoint_hashes": hashes,
        "seeds": list(seeds),
        "paired_block_size": NUM_PLAYERS,
        "games": games,
        "win_rates": summaries,
        "final_net_worth": net_worth,
        "truncations": sum(game["truncated"] for game in games),
        "scripted_compatibility_fallbacks": sum(
            sum(game["scripted_compatibility_fallbacks"]) for game in games
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "disclaimer": RESULTS_DISCLAIMER,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seat-balanced ppo-plus-v2 evaluation in four-game seed blocks"
    )
    parser.add_argument("--focus", default="asu-value-v1")
    parser.add_argument(
        "--opponents",
        nargs=3,
        default=("fixed-a", "fixed-b", "fixed-c"),
        metavar=("AGENT_A", "AGENT_B", "AGENT_C"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(0,),
        help="one seed per paired four-game block",
    )
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_lineup(
            args.focus,
            tuple(args.opponents),
            tuple(args.seeds),
            args.max_decisions,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    payload = json.dumps(
        result,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentFactory",
    "AgentSpec",
    "RESULTS_DISCLAIMER",
    "checkpoint_sha256",
    "evaluate_lineup",
    "main",
    "parse_agent_spec",
    "wilson_interval",
]
