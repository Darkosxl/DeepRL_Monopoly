"""Policy/value-guided stochastic multiplayer Max-N PUCT."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Mapping

import numpy as np

from .config import SearchConfig
from .contracts import SearchResult, Vector
from .engine import (
    DICE_OUTCOMES,
    NUM_PLAYERS,
    OFFSETS,
    ActionType,
    MonopolyEnv,
    SharedGame,
    action_family,
    clone_env,
    terminal_value,
    transition,
)
from .model import MonopolyZeroNet


def _vector(value: np.ndarray) -> Vector:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def progressive_width(legal_count: int, visits: int, config: SearchConfig) -> int:
    return min(
        config.max_width,
        legal_count,
        config.widening_base + math.floor(config.widening_scale * math.sqrt(visits)),
    )


def progressive_actions(
    legal_actions: tuple[int, ...],
    priors: Mapping[int, float],
    visits: int,
    config: SearchConfig,
) -> tuple[int, ...]:
    """Keep binary actions and each family's policy leader, then fill by prior."""
    if not legal_actions:
        raise ValueError("Progressive widening received no legal actions")
    missing = set(legal_actions) - set(priors)
    if missing:
        raise ValueError(f"Missing priors for legal actions: {sorted(missing)[:3]}")

    mandatory = {action for action in legal_actions if action < OFFSETS["mortgage"]}
    families: dict[str, list[int]] = {}
    for action in legal_actions:
        families.setdefault(action_family(action), []).append(action)
    mandatory.update(
        max(actions, key=lambda action: (priors[action], -action))
        for actions in families.values()
    )

    ordered = sorted(legal_actions, key=lambda action: (-priors[action], action))
    limit = max(progressive_width(len(legal_actions), visits, config), len(mandatory))
    selected = set(mandatory)
    for action in ordered:
        if len(selected) >= limit:
            break
        selected.add(action)
    return tuple(action for action in ordered if action in selected)


@dataclass(slots=True)
class OutcomeStats:
    probability: float = 1.0 / 36.0
    visits: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float64))
    child: "DecisionNode | None" = None

    @property
    def q(self) -> np.ndarray:
        return self.value_sum / self.visits if self.visits else np.zeros(NUM_PLAYERS)


@dataclass(slots=True)
class ChanceNode:
    outcomes: dict[tuple[int, int], OutcomeStats] = field(
        default_factory=lambda: {outcome: OutcomeStats() for outcome in DICE_OUTCOMES}
    )
    visits: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float64))

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        return DICE_OUTCOMES[int(rng.integers(len(DICE_OUTCOMES)))]


@dataclass(slots=True)
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float64))
    child: "DecisionNode | ChanceNode | None" = None

    @property
    def q(self) -> np.ndarray:
        return self.value_sum / self.visits if self.visits else np.zeros(NUM_PLAYERS)


@dataclass(slots=True)
class DecisionNode:
    env: MonopolyEnv
    actor_id: int
    legal_actions: tuple[int, ...]
    priors: dict[int, float]
    initial_value: np.ndarray
    edges: dict[int, EdgeStats] = field(default_factory=dict)
    visits: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float64))

    @property
    def mean_value(self) -> np.ndarray:
        return self.value_sum / self.visits if self.visits else self.initial_value


class MaxNPUCT:
    """Each decision node selects with the acting physical player's value."""

    def __init__(
        self,
        model: MonopolyZeroNet,
        config: SearchConfig | None = None,
        *,
        self_play: bool = False,
    ):
        self.model = model
        self.config = config or SearchConfig()
        self.self_play = self_play
        self.last_root: DecisionNode | None = None

    def _evaluate(self, env: MonopolyEnv) -> DecisionNode:
        actor = env.whose_turn()
        legal = tuple(env.get_allowed_actions(actor))
        priors, value = self.model.predict(env._get_state(actor), legal, actor)
        probabilities = np.asarray(tuple(priors.values()), dtype=np.float64)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError("Policy produced invalid legal priors")
        total = float(probabilities.sum())
        if total <= 0:
            raise RuntimeError("Policy assigned no probability to legal actions")
        priors = {action: probability / total for action, probability in priors.items()}
        return DecisionNode(env, actor, legal, priors, value.astype(np.float64))

    def _widen(self, node: DecisionNode) -> None:
        for action in progressive_actions(node.legal_actions, node.priors, node.visits, self.config):
            node.edges.setdefault(action, EdgeStats(node.priors[action]))

    def _select(self, node: DecisionNode) -> tuple[int, EdgeStats]:
        self._widen(node)
        scale = math.sqrt(max(node.visits, 1))
        actor = node.actor_id
        return max(
            node.edges.items(),
            key=lambda item: (
                float(item[1].q[actor])
                + self.config.c_puct * item[1].prior * scale / (1 + item[1].visits),
                item[1].prior,
                -item[0],
            ),
        )

    def _simulate(
        self,
        node: DecisionNode,
        depth: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if node.env.done:
            return terminal_value(node.env).astype(np.float64)
        if depth >= self.config.max_depth:
            node.visits += 1
            node.value_sum += node.initial_value
            return node.initial_value

        action, edge = self._select(node)
        if action == int(ActionType.ROLL_DICE):
            if edge.child is None:
                edge.child = ChanceNode()
            chance = edge.child
            if not isinstance(chance, ChanceNode):
                raise RuntimeError("Dice edge does not lead to a chance node")
            outcome = chance.sample(rng)
            outcome_stats = chance.outcomes[outcome]
            next_env = transition(node.env, action, outcome)
            if next_env.done:
                value = terminal_value(next_env).astype(np.float64)
            else:
                if outcome_stats.child is None:
                    outcome_stats.child = self._evaluate(next_env)
                    value = outcome_stats.child.initial_value
                    outcome_stats.child.visits += 1
                    outcome_stats.child.value_sum += value
                else:
                    value = self._simulate(outcome_stats.child, depth + 1, rng)
            outcome_stats.visits += 1
            outcome_stats.value_sum += value
            chance.visits += 1
            chance.value_sum += value
        else:
            next_env = transition(node.env, action)
            if next_env.done:
                value = terminal_value(next_env).astype(np.float64)
            else:
                if edge.child is None:
                    edge.child = self._evaluate(next_env)
                    value = edge.child.initial_value
                    edge.child.visits += 1
                    edge.child.value_sum += value
                elif isinstance(edge.child, ChanceNode):
                    raise RuntimeError("Non-dice edge unexpectedly leads to chance")
                else:
                    value = self._simulate(edge.child, depth + 1, rng)

        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        node.value_sum += value
        return value

    def _add_root_noise(self, root: DecisionNode, rng: np.random.Generator) -> None:
        actions = root.legal_actions
        noise = rng.dirichlet(np.full(len(actions), self.config.dirichlet_alpha))
        epsilon = self.config.dirichlet_epsilon
        root.priors = {
            action: (1 - epsilon) * root.priors[action] + epsilon * float(noise[index])
            for index, action in enumerate(actions)
        }

    def _final_action(self, root: DecisionNode, rng: np.random.Generator) -> int:
        if self.self_play and root.env.round <= self.config.temperature_rounds:
            actions = tuple(root.edges)
            counts = np.asarray([root.edges[action].visits for action in actions], dtype=np.float64)
            if self.config.visit_temperature != 1:
                counts = counts ** (1 / self.config.visit_temperature)
            probabilities = counts / counts.sum()
            return int(rng.choice(actions, p=probabilities))
        return max(
            root.edges,
            key=lambda action: (root.edges[action].visits, root.edges[action].prior, -action),
        )

    def choose_action(
        self,
        game: MonopolyEnv | SharedGame,
        player_id: int,
        decision_seed: int,
    ) -> SearchResult:
        """Search a non-forced decision without mutating the supplied game."""
        started = time.perf_counter()
        env = clone_env(game)
        if env.done:
            raise ValueError("Cannot choose an action in a finished game")
        if env.whose_turn() != player_id:
            raise ValueError(f"Player {player_id} is not the current actor")
        rng = np.random.default_rng(decision_seed)
        root = self._evaluate(env)
        self.last_root = root

        if len(root.legal_actions) == 1:
            action = root.legal_actions[0]
            return SearchResult(
                chosen_action=action,
                visits={action: 1},
                q_values={action: _vector(root.initial_value)},
                root_value=_vector(root.initial_value),
                simulations=0,
                latency_s=time.perf_counter() - started,
            )

        if self.self_play:
            self._add_root_noise(root, rng)
        for _ in range(self.config.simulations):
            self._simulate(root, 0, rng)
        action = self._final_action(root, rng)
        return SearchResult(
            chosen_action=action,
            visits={candidate: edge.visits for candidate, edge in root.edges.items()},
            q_values={candidate: _vector(edge.q) for candidate, edge in root.edges.items()},
            root_value=_vector(root.mean_value),
            simulations=self.config.simulations,
            latency_s=time.perf_counter() - started,
        )


PUCTAgent = MaxNPUCT
