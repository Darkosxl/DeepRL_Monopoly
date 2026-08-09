"""Shared PPO actor trunk with a four-player win-probability head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .engine import (
    ACTION_SPACE_SIZE,
    NUM_PLAYERS,
    RULESET_VERSION,
    STATE_DIM,
    relative_to_physical,
)
from monopoly_game_engine.networks import ActorNetwork


class MonopolyZeroNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        actor = ActorNetwork(hidden_dim)
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(*list(actor.net.children())[:-1])
        self.policy_head = list(actor.net.children())[-1]
        self.value_head = nn.Linear(hidden_dim, NUM_PLAYERS)

    def forward(
        self,
        states: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        logits = self.policy_head(features)
        if legal_mask is not None:
            if not legal_mask.any(dim=-1).all():
                raise ValueError("MonopolyZeroNet received an empty legal mask")
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits, torch.softmax(self.value_head(features), dim=-1)

    def load_ppo_actor(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 3,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "hidden_dim": self.hidden_dim,
        }
        actual = {name: payload.get(name) for name in expected}
        if actual != expected:
            raise ValueError(f"Incompatible PPO bootstrap: {actual}; expected {expected}")
        actor = ActorNetwork(self.hidden_dim)
        actor.load_state_dict(payload["actor"])
        self.trunk.load_state_dict(actor.net[:-1].state_dict())
        self.policy_head.load_state_dict(actor.net[-1].state_dict())
        return actual

    def freeze_policy(self) -> None:
        for parameter in (*self.trunk.parameters(), *self.policy_head.parameters()):
            parameter.requires_grad_(False)
        for parameter in self.value_head.parameters():
            parameter.requires_grad_(True)

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in (*self.policy_head.parameters(), *self.value_head.parameters()):
            parameter.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def predict(
        self,
        state: np.ndarray,
        legal_actions: Iterable[int],
        actor_id: int,
    ) -> tuple[dict[int, float], np.ndarray]:
        legal = tuple(int(action) for action in legal_actions)
        if not legal:
            raise ValueError("Cannot predict without a legal action")
        device = next(self.parameters()).device
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros((1, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
        mask[0, list(legal)] = True
        with torch.inference_mode():
            logits, relative_value = self(state_tensor, mask)
            probabilities = torch.softmax(logits, dim=-1)[0]
        priors = {action: float(probabilities[action]) for action in legal}
        value = relative_to_physical(relative_value[0].cpu().numpy(), actor_id)
        return priors, value

    def save_inference(self, path: str | Path, metadata: dict | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": 1,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "players": NUM_PLAYERS,
                "hidden_dim": self.hidden_dim,
                "model": self.state_dict(),
                "metadata": metadata or {},
            },
            temporary,
        )
        os.replace(temporary, destination)

    @classmethod
    def load_inference(cls, path: str | Path, device: str | torch.device = "cpu") -> "MonopolyZeroNet":
        payload = torch.load(path, map_location=device, weights_only=True)
        expected = (RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE, NUM_PLAYERS)
        actual = tuple(payload.get(name) for name in ("ruleset", "state_dim", "action_dim", "players"))
        if actual != expected:
            raise ValueError(f"Incompatible MonopolyZero model: {actual}; expected {expected}")
        model = cls(int(payload.get("hidden_dim", 256))).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model
