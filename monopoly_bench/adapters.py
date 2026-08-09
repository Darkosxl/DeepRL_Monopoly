"""Policies for the six fixed agents, PPO, CFR, search, and callable SLMs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
import random
import sys
import time
from typing import Callable

import numpy as np
import torch

from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1

from .config import SearchConfig
from .contracts import SearchResult
from .engine import (
    ACTION_SPACE_SIZE,
    ROOT,
    RULESET_VERSION,
    STATE_DIM,
    ActionType,
    unwrap,
)
from .model import MonopolyZeroNet
from .search import MaxNPUCT
from monopoly_game_engine.agent_ppo import fixed_accept_trade_decision, fixed_buy_decision
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
from monopoly_game_engine.networks import ActorNetwork


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: int
    latency_s: float
    fallback: bool = False
    error: str | None = None


class SearchAdapter:
    def __init__(self, model: MonopolyZeroNet, config: SearchConfig, *, self_play: bool):
        self.search = MaxNPUCT(model, config, self_play=self_play)

    def choose_action(self, game, player_id: int, decision_seed: int) -> SearchResult:
        return self.search.choose_action(game, player_id, decision_seed)


class FixedAdapter:
    def __init__(self, agent_class):
        if agent_class not in FP_AGENT_CLASSES:
            raise ValueError(f"Unknown fixed personality: {agent_class}")
        self.agent_class = agent_class
        self.name = agent_class.__name__
        self.compatibility_fallbacks = 0

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        started = time.perf_counter()
        env = unwrap(game)
        outer = random.getstate()
        try:
            random.seed(decision_seed)
            action = int(self.agent_class(player_id).choose_action(env))
        finally:
            random.setstate(outer)
        legal = tuple(env.get_allowed_actions(player_id))
        if action not in legal:
            self.compatibility_fallbacks += 1
            action = (
                int(ActionType.END_TURN)
                if int(ActionType.END_TURN) in legal
                else legal[0]
            )
        return ActionDecision(action, time.perf_counter() - started)


class ASUAdapter:
    """Bind either frozen ASU policy to the benchmark policy contract."""

    def __init__(self, *, rollout: bool = False):
        self.agent_class = ASURolloutV1 if rollout else ASUValueV1
        self.name = "asu_rollout_v1" if rollout else "asu_value_v1"
        self._agents = {}

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        del decision_seed
        started = time.perf_counter()
        env = unwrap(game)
        agent = self._agents.setdefault(player_id, self.agent_class(player_id))
        action = int(agent.choose_action(env))
        if action not in env.get_allowed_actions(player_id):
            raise RuntimeError(f"{self.name} produced illegal action {action}")
        return ActionDecision(action, time.perf_counter() - started)


class PPOAdapter:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        stochastic: bool = False,
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        actual = tuple(payload.get(name) for name in ("ruleset", "state_dim", "action_dim"))
        expected = (RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE)
        if actual != expected:
            raise ValueError(f"Incompatible PPO baseline: {actual}; expected {expected}")
        self.actor = ActorNetwork(int(payload.get("hidden_dim", 256))).to(device)
        self.actor.load_state_dict(payload["actor"])
        self.actor.eval()
        self.hybrid = bool(payload.get("hybrid", False))
        self.stochastic = stochastic
        self.name = "ppo_v2"

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        started = time.perf_counter()
        env = unwrap(game)
        legal = tuple(env.get_allowed_actions(player_id))
        if self.hybrid and int(ActionType.BUY_PROPERTY) in legal and fixed_buy_decision(env, player_id):
            return ActionDecision(int(ActionType.BUY_PROPERTY), time.perf_counter() - started)
        if self.hybrid and int(ActionType.ACCEPT_TRADE) in legal:
            action = (
                int(ActionType.ACCEPT_TRADE)
                if fixed_accept_trade_decision(env, player_id)
                else int(ActionType.DECLINE_TRADE)
            )
            return ActionDecision(action, time.perf_counter() - started)

        neural_legal = tuple(
            action
            for action in legal
            if not self.hybrid or action not in (int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE))
        )
        if not neural_legal:
            raise RuntimeError("PPO baseline has no neural legal action")
        device = next(self.actor.parameters()).device
        state = torch.as_tensor(env._get_state(player_id), dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros((1, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
        mask[0, list(neural_legal)] = True
        with torch.inference_mode():
            probabilities = self.actor(state, mask).exp()[0]
        if self.stochastic:
            weights = probabilities[list(neural_legal)].cpu().numpy().astype(np.float64)
            weights /= weights.sum()
            action = int(np.random.default_rng(decision_seed).choice(neural_legal, p=weights))
        else:
            action = max(neural_legal, key=lambda candidate: (float(probabilities[candidate]), -candidate))
        return ActionDecision(action, time.perf_counter() - started)


class CFRAdapter:
    def __init__(self, checkpoint: str | Path, *, stochastic: bool = False):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = importlib.import_module(
            "RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.classic_cfr"
        )
        self._shared_game = module.SharedGame
        self.model = module.MonteCarloCFR.load(checkpoint)
        self.stochastic = stochastic
        self.name = "cfr_v2"

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        started = time.perf_counter()
        env = unwrap(game)
        shared = self._shared_game(copy.deepcopy(env), random.Random(decision_seed).getstate())
        legal = shared.legal_actions()
        node = self.model.tables[player_id].get(shared.information_key(player_id))
        probabilities = (
            node.average_policy()
            if node is not None
            else np.full(len(legal), 1 / len(legal), dtype=np.float32)
        )
        if self.stochastic:
            action = int(np.random.default_rng(decision_seed).choice(legal, p=probabilities))
        else:
            action = max(zip(legal, probabilities), key=lambda pair: (float(pair[1]), -pair[0]))[0]
        return ActionDecision(int(action), time.perf_counter() - started)


class SLMAdapter:
    """Expose an existing Gemma-style ``generate(prompt)`` callable as a policy."""

    def __init__(self, generate: Callable[[str], str]):
        self.generate = generate
        if str(ROOT / "SLM_HANDMADE_MONOPOLY") not in sys.path:
            sys.path.insert(0, str(ROOT / "SLM_HANDMADE_MONOPOLY"))
        contract = importlib.import_module("monopoly_qlora")
        self._prompt = contract.canonical_prompt
        self._parse = contract.parse_action_json
        self._fallback = contract.fallback_action
        self.name = "slm_callable"

    def choose_action(self, game, player_id: int, decision_seed: int) -> ActionDecision:
        del decision_seed
        started = time.perf_counter()
        env = unwrap(game)
        legal = tuple(env.get_allowed_actions(player_id))
        raw = self.generate(self._prompt(env, player_id))
        try:
            action = self._parse(raw, env, player_id, legal)
            return ActionDecision(action, time.perf_counter() - started)
        except ValueError as exc:
            return ActionDecision(
                self._fallback(legal),
                time.perf_counter() - started,
                fallback=True,
                error=str(exc),
            )


def fixed_baselines() -> dict[str, FixedAdapter]:
    return {agent_class.__name__: FixedAdapter(agent_class) for agent_class in FP_AGENT_CLASSES}


def available_baselines(
    ppo_checkpoint: str | Path | None = None,
    cfr_checkpoint: str | Path | None = None,
    *,
    include_asu_rollout: bool = False,
) -> dict[str, object]:
    ppo_path = Path(ppo_checkpoint or ROOT / "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt")
    cfr_path = Path(cfr_checkpoint or ROOT / "artifacts/cfr_ppo_plus/cfr_full_game_v2.pkl.gz")
    baselines: dict[str, object] = fixed_baselines()
    baselines["asu_value_v1"] = ASUAdapter()
    if include_asu_rollout:
        baselines["asu_rollout_v1"] = ASUAdapter(rollout=True)
    if ppo_path.exists():
        baselines["ppo_v2"] = PPOAdapter(ppo_path)
    if cfr_path.exists():
        baselines["cfr_v2"] = CFRAdapter(cfr_path)
    return baselines
