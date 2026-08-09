"""
PPO Agent (Section V-A-1 and V-B).

Implements actor-critic PPO with clipped surrogate objective and
truncated GAE advantage estimation, as described in the paper.

Hybrid mode: BUY_PROPERTY and ACCEPT_TRADE are handled by fixed rules,
             all other actions go through the actor network.

Fixes applied
─────────────
Bug 1 – Hybrid actions no longer stored in the PPO buffer.
    choose_action() now returns a sentinel log_prob=None to signal that
    the action was chosen by the fixed policy and must not be stored.
    train.py should check `if log_prob is not None` before calling store().

Bug 2 – Correct action mask used during update().
    Each transition now records the allowed_actions mask at collection
    time. The PPO update uses that per-step mask instead of an all-ones
    mask, so the actor is never trained on actions that were illegal in
    the state where the transition was collected.

Bug 3 – Proper GAE bootstrap for mid-game rollout boundaries.
    _compute_gae() now accepts an explicit `last_next_value` argument.
    When the buffer is flushed mid-game (done=False at the boundary),
    train.py should query the critic on the next state and pass that
    value here so the advantage estimate is not truncated to zero.
"""

from typing import List, Optional
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actions import ACTION_SPACE_SIZE, OFFSETS, ActionType
from .constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    NUM_PLAYERS,
    PROPERTY_IDS,
    RULESET_VERSION,
    TRADE_CASH_LEVELS,
)
from .networks import ActorNetwork, CriticNetwork
from .state import STATE_DIM

# ── Hybrid fixed-policy decisions ─────────────────────────────────────────────


def fixed_buy_decision(env, pid: int) -> bool:
    player = env.players[pid]
    sq = player.position
    if sq not in env.properties:
        return False
    prop = env.properties[sq]
    if prop.owner is not None or not player.can_afford(prop.price):
        return False

    # Always buy if it completes a monopoly
    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    if group:
        owned = sum(1 for s in group if env.properties[s].owner == pid)
        if owned + 1 == len(group):
            return True

    return player.cash >= prop.price + 100


def fixed_accept_trade_decision(env, pid: int) -> bool:
    offer = env._incoming_trade(pid)
    if offer is None:
        return False

    if offer.offered_prop:
        color = offer.offered_prop.color
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned_after = sum(
                1
                for s in group
                if env.properties[s].owner == pid
                or env.properties[s] == offer.offered_prop
            )
            if owned_after == len(group):
                return True

    nwo = offer.net_worth()
    return nwo >= 0


# ── Experience buffer ─────────────────────────────────────────────────────────


class PPOBuffer:
    """Stores a single rollout trajectory for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []  # FIX 2: per-step allowed-action masks

    def store(self, state, action, log_prob, reward, value, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)  # FIX 2

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── PPO Agent ─────────────────────────────────────────────────────────────────


class PPOAgent:
    """
    Actor-critic PPO agent, with optional hybrid mode.

    Parameters mirror Appendix B-A of the paper.
    """

    def __init__(
        self,
        player_id: int,
        hybrid: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 1024,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 256,
        win_loss_bonus: float = 1.0,
        device: str = "auto",
    ):
        self.player_id = player_id
        self.hybrid = hybrid
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.win_loss_bonus = win_loss_bonus
        self.hidden_dim = hidden_dim
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        self.actor = ActorNetwork(hidden_dim).to(self.device)
        self.critic = CriticNetwork(hidden_dim).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = PPOBuffer()
        self.step_count = 0
        self.games_trained = 0

        # Mask actions permanently handled by fixed policy (hybrid only)
        self.fixed_action_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        if hybrid:
            self.fixed_action_mask[int(ActionType.BUY_PROPERTY)] = True
            self.fixed_action_mask[int(ActionType.ACCEPT_TRADE)] = True

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray, env, allowed_actions: List[int]):
        """
        Choose an action. In hybrid mode, intercept BUY and ACCEPT_TRADE.

        Returns (action, log_prob, value, allowed_actions).

        FIX 1: When the hybrid fixed policy fires, log_prob is returned as
        None.  The caller (train.py) must NOT store these transitions in the
        PPO buffer — storing them with log_prob=0 corrupts the importance
        sampling ratio.
        """
        pid = self.player_id

        # Hybrid: handle buy property
        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed_actions:
            if fixed_buy_decision(env, pid):
                # FIX 1: return None sentinel — caller skips buffer storage
                return int(ActionType.BUY_PROPERTY), None, None, allowed_actions

        # Hybrid: handle trade acceptance
        if self.hybrid and int(ActionType.ACCEPT_TRADE) in allowed_actions:
            pending = next(
                (o for o in env.pending_trades.values() if o.to_player == pid),
                None,
            )
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE), None, None, allowed_actions
                else:
                    return int(ActionType.DECLINE_TRADE), None, None, allowed_actions

        # Filter out fixed-policy actions from neural net consideration
        nn_allowed = [a for a in allowed_actions if not self.fixed_action_mask[a]]
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            value = self.critic(state_t).item()
        action, log_prob = self.actor.get_action(state, nn_allowed)

        return action, log_prob, value, nn_allowed  # FIX 2: return nn_allowed

    # ── Store experience ──────────────────────────────────────────────────────

    def store(
        self, state, action, log_prob, reward, value, done, allowed_actions: List[int]
    ):
        """
        Store one NN-chosen transition.
        FIX 1: Callers must only call this when log_prob is not None (i.e.
               the action was chosen by the neural network, not the fixed
               hybrid policy).
        FIX 2: allowed_actions is stored so update() can reconstruct the
               per-step action mask.
        """
        # Build boolean mask for the allowed actions at this step (FIX 2)
        mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[allowed_actions] = True

        self.buffer.store(state, action, log_prob, reward, value, done, mask)
        self.step_count += 1

    def add_win_loss(self, won: bool):
        """Add terminal win/loss bonus to the last stored reward."""
        if self.win_loss_bonus != 0 and len(self.buffer.rewards) > 0:
            self.buffer.rewards[-1] += self.win_loss_bonus * (1.0 if won else -1.0)

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self, last_next_state: Optional[np.ndarray] = None, last_done: bool = False
    ):
        """
        Run PPO update over the current buffer.

        FIX 3: last_next_state / last_done support correct GAE bootstrap.
            Pass the state that followed the last stored transition and
            whether that transition was terminal.  When the rollout ends
            mid-game (last_done=False), the critic evaluates last_next_state
            to get the bootstrap value instead of using 0.
        """
        if len(self.buffer) == 0:
            return {}

        # FIX 3: bootstrap value for the end of the rollout
        if last_next_state is not None and not last_done:
            with torch.inference_mode():
                st = torch.as_tensor(
                    last_next_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                bootstrap_value = self.critic(st).item()
        else:
            bootstrap_value = 0.0

        # Convert to tensors
        states = torch.as_tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_lps = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        # FIX 2: per-step masks stacked into (N, ACTION_SPACE_SIZE)
        step_masks = torch.stack(self.buffer.action_masks).to(self.device)

        # FIX 3: pass bootstrap value into GAE
        advantages = self._compute_gae(rewards, values, dones, bootstrap_value)
        returns = advantages + torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue

                sb = states[idx]
                ab = actions[idx]
                olp = old_lps[idx]
                adv = advantages[idx]
                ret = returns[idx]
                mask = step_masks[idx]  # FIX 2: per-step masks for this batch

                # FIX 2: use the recorded per-step mask, not all-ones
                log_probs_all = self.actor(sb, mask)
                new_lps = log_probs_all.gather(1, ab.unsqueeze(1)).squeeze(1)
                probs_all = log_probs_all.exp()
                # Masked actions have log_prob=-inf and prob=0. Multiplying
                # them directly yields 0 * -inf = nan, which can poison the
                # PPO loss and corrupt the actor weights.
                safe_log_probs = torch.where(
                    mask, log_probs_all, torch.zeros_like(log_probs_all)
                )
                entropy = -(probs_all * safe_log_probs).sum(dim=-1).mean()

                ratio = (new_lps - olp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(sb)
                critic_loss = nn.MSELoss()(values_pred, ret)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.opt.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                n_batches += 1

        self.buffer.clear()
        if n_batches > 0:
            return {k: v / n_batches for k, v in stats.items()}
        return stats

    def _compute_gae(
        self, rewards, values, dones, last_next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Generalised Advantage Estimation.

        FIX 3: last_next_value is the critic's estimate of V(s_{T+1}).
        For terminal transitions it is 0; for mid-game buffer flushes it is
        the critic's evaluation of the state that followed the last stored
        step, preventing the advantage from being truncated to zero.
        """
        advantages = torch.zeros(len(rewards), device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t + 1 < len(values):
                next_val = values[t + 1]
            else:
                # FIX 3: use bootstrapped value, not hard-coded 0
                next_val = last_next_value
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def save(self, path: str):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": 3,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "player_id": self.player_id,
                "hybrid": self.hybrid,
                "hidden_dim": self.hidden_dim,
                "step_count": self.step_count,
                "games_trained": self.games_trained,
                "training_config": {
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "clip_eps": self.clip_eps,
                    "entropy_coef": self.entropy_coef,
                    "value_coef": self.value_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "n_steps": self.n_steps,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "win_loss_bonus": self.win_loss_bonus,
                },
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, destination)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get("format_version") is None:
            raise ValueError(
                "Legacy PPO checkpoint is incompatible with "
                f"{RULESET_VERSION}; train a new checkpoint."
            )
        expected = {
            "format_version": 3,
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
                f"Incompatible PPO checkpoint metadata: {actual}; "
                f"expected {expected}."
            )
        try:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
        except RuntimeError as exc:
            raise ValueError(
                f"PPO checkpoint network is incompatible with {RULESET_VERSION}."
            ) from exc
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
            for state in self.opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        for key, value in ckpt.get("training_config", {}).items():
            setattr(self, key, value)
        self.step_count = int(ckpt.get("step_count", 0))
        self.games_trained = int(ckpt.get("games_trained", 0))
