"""
Neural network architectures for PPO (actor-critic) and DDQN agents.
State dim = STATE_DIM, Action dim = ACTION_SPACE_SIZE

Fix 4 applied
─────────────
The original 6-layer network caused
vanishing gradients: with plain ReLU and no normalisation the early layers
received near-zero gradients throughout training.

Changes:
  - Depth reduced to 3 hidden layers.
  - nn.LayerNorm added after every hidden activation.  LayerNorm is
    preferred over BatchNorm here because rollout batch sizes can be small
    and variable; LayerNorm operates per-sample and is unaffected by batch
    size.
  - hidden_dim default changed from 512 to 256 to match the shallower
    architecture.  Callers that previously passed hidden_dim=256 to
    PPOAgent will get an identical result; callers that passed hidden_dim=512
    will now get a wider-but-still-shallow network which is fine.

DDQN uses the paper's separate 1024→512 ReLU architecture.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import ACTION_SPACE_SIZE
from .state import STATE_DIM


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    """Linear → LayerNorm → ReLU block."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),  # FIX 4: normalise before activation
        nn.ReLU(),
    )


class ActorNetwork(nn.Module):
    """
    PPO Actor: maps state → action log-probability distribution.

    Architecture (FIX 4 – shallower + LayerNorm):
        STATE_DIM → [256 → LN → ReLU] × 3 → ACTION_SPACE_SIZE
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_block(STATE_DIM, hidden_dim),
            _mlp_block(hidden_dim, hidden_dim),
            _mlp_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, ACTION_SPACE_SIZE),
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            state : (batch, STATE_DIM)
            mask  : (batch, ACTION_SPACE_SIZE) bool tensor, True = allowed
        Returns:
            log_probs : (batch, ACTION_SPACE_SIZE)
        """
        logits = self.net(state)
        if mask is not None:
            if not mask.any(dim=-1).all():
                raise ValueError("ActorNetwork received an action mask with no valid actions")
            logits = logits.masked_fill(~mask, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def get_action(self, state: np.ndarray, allowed_actions: list):
        """Sample an action given allowed actions."""
        if len(allowed_actions) == 0:
            raise ValueError("ActorNetwork.get_action() received no allowed actions")
        if not np.isfinite(state).all():
            raise ValueError("ActorNetwork.get_action() received a non-finite state")

        device = next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
        mask[0, allowed_actions] = True
        with torch.inference_mode():
            log_probs = self.forward(state_t, mask)
        probs = log_probs.exp().squeeze(0)
        if not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
            raise RuntimeError(
                "ActorNetwork produced an invalid action distribution; "
                "check PPO updates for NaNs/Infs"
            )
        action = torch.multinomial(probs, 1).item()
        log_prob = log_probs[0, action].item()
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    PPO Critic: maps state → scalar value estimate V(s).

    Architecture (FIX 4 – shallower + LayerNorm):
        STATE_DIM → [256 → LN → ReLU] × 3 → 1
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_block(STATE_DIM, hidden_dim),
            _mlp_block(hidden_dim, hidden_dim),
            _mlp_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class DDQNNetwork(nn.Module):
    """
    Double DQN Q-network from Appendix B-B of the paper.

    Architecture:
        STATE_DIM → 1024 → ReLU → 512 → ReLU → ACTION_SPACE_SIZE
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, ACTION_SPACE_SIZE),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def get_action(
        self, state: np.ndarray, allowed_actions: list, epsilon: float = 0.0
    ):
        """ε-greedy action selection with action masking."""
        if random.random() < epsilon:
            return random.choice(allowed_actions)
        device = next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.forward(state_t).squeeze(0)
        mask = torch.full(
            (ACTION_SPACE_SIZE,), float("-inf"), device=device
        )
        mask[allowed_actions] = 0.0
        q_masked = q_values + mask
        return q_masked.argmax().item()
