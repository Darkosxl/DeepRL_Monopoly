"""
Double DQN (DDQN) Agent (Section V-A-2 and V-B).

Uses:
  - Experience replay buffer
  - Target network with periodic hard updates
  - ε-greedy exploration with exponential decay
  - Action masking to only consider valid actions

Hybrid mode: BUY_PROPERTY and ACCEPT_TRADE handled by fixed rules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple

from .networks import DDQNNetwork
from .actions import ActionType, ACTION_SPACE_SIZE
from .constants import COLOR_GROUPS, NUM_PLAYERS
from .agent_ppo import fixed_buy_decision, fixed_accept_trade_decision


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

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
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,  # exponential decay per step
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,  # steps between target network updates
        hidden_dim: int = 256,
        win_loss_bonus: float = 10.0,   # constant c=10 for DDQN (paper Exp 1)
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

        self.online_net = DDQNNetwork(hidden_dim)
        self.target_net = DDQNNetwork(hidden_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self.step_count = 0
        self.last_state  = None
        self.last_action = None

        # For hybrid: permanently mask these from the neural net
        self.fixed_actions = set()
        if hybrid:
            self.fixed_actions.add(int(ActionType.BUY_PROPERTY))
            self.fixed_actions.add(int(ActionType.ACCEPT_TRADE))

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray, env, allowed_actions: List[int]) -> int:
        pid = self.player_id

        # Hybrid: intercept buy
        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed_actions:
            if fixed_buy_decision(env, pid):
                return int(ActionType.BUY_PROPERTY)

        # Hybrid: intercept trade acceptance
        if self.hybrid:
            pending = next(
                (o for o in env.pending_trades.values() if o.to_player == pid),
                None
            )
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE)
                return int(ActionType.DECLINE_TRADE)

        # NN actions only
        nn_allowed = [a for a in allowed_actions if a not in self.fixed_actions]
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]

        action = self.online_net.get_action(state, nn_allowed, self.epsilon)
        return action

    # ── Learning step ─────────────────────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def add_win_loss(self, won: bool):
        """Add win/loss bonus to the most recent transition."""
        if self.win_loss_bonus != 0 and len(self.buffer) > 0:
            s, a, r, ns, d = self.buffer.buffer[-1]
            bonus = self.win_loss_bonus if won else -self.win_loss_bonus
            self.buffer.buffer[-1] = (s, a, r + bonus, ns, d)

    def update(self) -> dict:
        """Sample mini-batch and perform a DDQN gradient step."""
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Hard update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.target_net.eval()
        self.epsilon = self.epsilon_end  # inference mode
