"""
Double DQN (DDQN) Agent (Section V-A-2 and V-B).

Uses:
  - Experience replay buffer
  - Target network with periodic hard updates
  - ε-greedy exploration with exponential decay
  - Action masking to only consider valid actions

Hybrid mode: BUY_PROPERTY and ACCEPT_TRADE handled by fixed rules.
"""

import os
import random
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import DDQNNetwork
from .actions import ACTION_SPACE_SIZE, OFFSETS, ActionType
from .constants import RULESET_VERSION
from .agent_ppo import fixed_buy_decision, fixed_accept_trade_decision
from .state import STATE_DIM


CHECKPOINT_VERSION = 3


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_allowed):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
                tuple(int(a) for a in next_allowed),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_allowed = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            next_allowed,
        )

    def state_dict(self) -> dict:
        items = list(self.buffer)
        states = (
            np.stack([x[0] for x in items])
            if items
            else np.empty((0, STATE_DIM), dtype=np.float32)
        )
        next_states = (
            np.stack([x[3] for x in items])
            if items
            else np.empty((0, STATE_DIM), dtype=np.float32)
        )
        return {
            "capacity": self.buffer.maxlen,
            "states": torch.from_numpy(states),
            "actions": torch.tensor([x[1] for x in items], dtype=torch.long),
            "rewards": torch.tensor([x[2] for x in items], dtype=torch.float32),
            "next_states": torch.from_numpy(next_states),
            "dones": torch.tensor([x[4] for x in items], dtype=torch.bool),
            "next_allowed": [list(x[5]) for x in items],
        }

    def load_state_dict(self, payload: dict) -> None:
        self.buffer = deque(maxlen=int(payload["capacity"]))
        for state, action, reward, next_state, done, next_allowed in zip(
            payload["states"].numpy(),
            payload["actions"].tolist(),
            payload["rewards"].tolist(),
            payload["next_states"].numpy(),
            payload["dones"].tolist(),
            payload["next_allowed"],
        ):
            self.push(state, action, reward, next_state, done, next_allowed)

    def __len__(self):
        return len(self.buffer)


# ── DDQN Agent ────────────────────────────────────────────────────────────────

class DDQNAgent:
    """
    Double DQN agent with experience replay and target network.
    
    Parameters from Appendix B-B of the paper.
    """

    def __init__(
        self,
        player_id: int,
        hybrid: bool = False,
        lr: float = 1e-5,
        gamma: float = 0.9999,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,  # exponential decay per game
        buffer_capacity: int = 10_000,
        batch_size: int = 128,
        target_update_freq: int = 500,  # games between target network updates
        hidden_dim: int = 1024,
        win_loss_bonus: float = 10.0,   # constant c=10 for DDQN (paper Exp 1)
        exploration_mode: str = "section_balanced",
        decision_penalty: float = 0.002,
        device: str = "auto",
    ):
        self.player_id       = player_id
        self.hybrid          = hybrid
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.batch_size      = batch_size
        self.target_update_freq = target_update_freq
        self.win_loss_bonus  = win_loss_bonus
        if exploration_mode not in {"section_balanced", "uniform_actions"}:
            raise ValueError(f"Unknown DDQN exploration mode: {exploration_mode}")
        self.exploration_mode = exploration_mode
        self.decision_penalty = decision_penalty
        self.hidden_dim = hidden_dim
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        self.online_net = DDQNNetwork(hidden_dim).to(self.device)
        self.target_net = DDQNNetwork(hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self.step_count = 0
        self.games_trained = 0

        # For hybrid: permanently mask these from the neural net
        self.fixed_actions = set()
        if hybrid:
            self.fixed_actions.add(int(ActionType.BUY_PROPERTY))
            self.fixed_actions.add(int(ActionType.ACCEPT_TRADE))

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray, env, allowed_actions: List[int]) -> int:
        return self.choose_training_action(state, env, allowed_actions)[0]

    def choose_training_action(
        self, state: np.ndarray, env, allowed_actions: List[int]
    ) -> tuple[int, tuple[int, ...] | None]:
        """Return the action and neural mask, or ``None`` for fixed decisions."""
        pid = self.player_id

        # Hybrid: intercept buy
        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed_actions:
            if fixed_buy_decision(env, pid):
                return int(ActionType.BUY_PROPERTY), None

        # Hybrid: intercept trade acceptance
        if self.hybrid and int(ActionType.ACCEPT_TRADE) in allowed_actions:
            pending = env._incoming_trade(pid)
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE), None
                return int(ActionType.DECLINE_TRADE), None

        # NN actions only
        nn_allowed = [a for a in allowed_actions if a not in self.fixed_actions]
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]

        if (
            self.exploration_mode == "section_balanced"
            and random.random() < self.epsilon
        ):
            action = self._balanced_random_action(nn_allowed)
        else:
            epsilon = self.epsilon if self.exploration_mode == "uniform_actions" else 0.0
            action = self.online_net.get_action(state, nn_allowed, epsilon)
        return action, tuple(nn_allowed)

    @staticmethod
    def _balanced_random_action(allowed_actions: List[int]) -> int:
        """Sample an action section first, then one legal action within it."""
        starts = sorted(OFFSETS.items(), key=lambda item: item[1], reverse=True)
        groups = {}
        for action in allowed_actions:
            section = next(name for name, start in starts if action >= start)
            groups.setdefault(section, []).append(action)
        return random.choice(random.choice(list(groups.values())))

    # ── Learning step ─────────────────────────────────────────────────────────

    def store_transition(
        self, state, action, reward, next_state, done, next_allowed
    ):
        if not done and not next_allowed:
            raise ValueError("Non-terminal DDQN transitions need legal next actions")
        self.buffer.push(state, action, reward, next_state, done, next_allowed)
        self.step_count += 1

    def finish_episode(self) -> None:
        self.games_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.games_trained % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def add_win_loss(self, won: bool):
        """Add win/loss bonus to the most recent transition."""
        if self.win_loss_bonus != 0 and len(self.buffer) > 0:
            s, a, r, ns, d, allowed = self.buffer.buffer[-1]
            bonus = self.win_loss_bonus if won else -self.win_loss_bonus
            self.buffer.buffer[-1] = (s, a, r + bonus, ns, d, allowed)

    @staticmethod
    def _masked_argmax(
        q_values: torch.Tensor, allowed_actions: tuple[tuple[int, ...], ...]
    ) -> torch.Tensor:
        mask = torch.zeros_like(q_values, dtype=torch.bool)
        for row, actions in enumerate(allowed_actions):
            valid = list(actions) if actions else [int(ActionType.DO_NOTHING)]
            mask[row, valid] = True
        return q_values.masked_fill(~mask, float("-inf")).argmax(1)

    def update(self) -> dict:
        """Sample mini-batch and perform a DDQN gradient step."""
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones, next_allowed = (
            self.buffer.sample(self.batch_size)
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self._masked_argmax(
                self.online_net(next_states), next_allowed
            )
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": CHECKPOINT_VERSION,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "player_id": self.player_id,
                "hybrid": self.hybrid,
                "hidden_dim": self.hidden_dim,
                "step_count": self.step_count,
                "games_trained": self.games_trained,
                "epsilon": self.epsilon,
                "training_config": {
                    "gamma": self.gamma,
                    "epsilon_end": self.epsilon_end,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "target_update_freq": self.target_update_freq,
                    "win_loss_bonus": self.win_loss_bonus,
                    "exploration_mode": self.exploration_mode,
                    "decision_penalty": self.decision_penalty,
                },
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay": self.buffer.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, destination)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format_version") is None:
            raise ValueError(
                "Legacy DDQN checkpoint is incompatible with "
                f"{RULESET_VERSION}; train a new checkpoint."
            )
        expected = {
            "format_version": CHECKPOINT_VERSION,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "player_id": self.player_id,
            "hybrid": self.hybrid,
            "hidden_dim": self.hidden_dim,
        }
        actual = {key: ckpt.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Incompatible DDQN checkpoint metadata: {actual}; "
                f"expected {expected}."
            )
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.target_net.eval()
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        training_config = dict(ckpt["training_config"])
        self.exploration_mode = training_config.pop(
            "exploration_mode", "uniform_actions"
        )
        self.decision_penalty = float(training_config.pop("decision_penalty", 0.0))
        for key, value in training_config.items():
            setattr(self, key, value)
        self.buffer.load_state_dict(ckpt["replay"])
        self.step_count = int(ckpt["step_count"])
        self.games_trained = int(ckpt["games_trained"])
        self.epsilon = float(ckpt["epsilon"])
