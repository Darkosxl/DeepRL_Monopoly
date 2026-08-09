from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from monopoly_bench.config import SearchConfig
from monopoly_bench.engine import (
    DICE_OUTCOMES,
    OFFSETS,
    ActionType,
    SharedGame,
    physical_to_relative,
    relative_to_physical,
    transition,
)
from monopoly_bench.model import MonopolyZeroNet
from monopoly_bench.search import (
    ChanceNode,
    DecisionNode,
    EdgeStats,
    MaxNPUCT,
    progressive_actions,
    progressive_width,
)
from monopoly_game_engine.networks import ActorNetwork


PPO = "artifacts/ppo_plus/ppo_hybrid_2000_v2.pt"


def test_all_36_ordered_dice_are_isolated() -> None:
    game = SharedGame.new(5, max_rounds=2)
    game.step(int(ActionType.END_TURN))
    outer = random.getstate()
    branches = [transition(game, int(ActionType.ROLL_DICE), outcome) for outcome in DICE_OUTCOMES]
    assert len(DICE_OUTCOMES) == len(set(DICE_OUTCOMES)) == 36
    assert [branch.last_dice for branch in branches] == list(DICE_OUTCOMES)
    assert random.getstate() == outer


def test_shared_clones_own_their_rng() -> None:
    outer = random.getstate()
    game = SharedGame.new(9, max_rounds=2)
    clone = game.clone()
    for copy_ in (game, clone):
        copy_.step(int(ActionType.END_TURN))
        copy_.step(int(ActionType.ROLL_DICE))
    assert game.env.last_dice == clone.env.last_dice
    assert game.env.players[game.env.active_player_id()].position == clone.env.players[clone.env.active_player_id()].position
    assert random.getstate() == outer


def test_actor_relative_vectors_round_trip() -> None:
    relative = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    physical = relative_to_physical(relative, actor_id=2)
    assert physical == pytest.approx([0.2, 0.3, 0.1, 0.4])
    assert physical_to_relative(physical, actor_id=2) == pytest.approx(relative)


def test_ppo_policy_logits_are_bit_exact_after_warm_start() -> None:
    payload = torch.load(PPO, map_location="cpu", weights_only=True)
    original = ActorNetwork()
    original.load_state_dict(payload["actor"])
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    states = torch.randn(7, 300)
    assert torch.equal(model(states)[0], original.net(states))


def test_progressive_widening_retains_binary_and_family_leaders() -> None:
    legal = (1, 2, OFFSETS["mortgage"], OFFSETS["mortgage"] + 1, OFFSETS["buy_trade"], OFFSETS["buy_trade"] + 1)
    priors = {action: (index + 1) / 21 for index, action in enumerate(legal)}
    chosen = progressive_actions(legal, priors, 0, SearchConfig())
    assert {1, 2}.issubset(chosen)
    assert OFFSETS["mortgage"] + 1 in chosen
    assert OFFSETS["buy_trade"] + 1 in chosen
    assert progressive_width(1_000, 196, SearchConfig()) == 64


def test_max_n_uses_the_acting_players_component() -> None:
    game = SharedGame.new(2, max_rounds=1).env
    model = MonopolyZeroNet()
    search = MaxNPUCT(model, SearchConfig(simulations=1))
    node = DecisionNode(
        game,
        actor_id=2,
        legal_actions=(1, 2),
        priors={1: 0.5, 2: 0.5},
        initial_value=np.full(4, 0.25),
        edges={
            1: EdgeStats(0.5, visits=2, value_sum=np.asarray([0.2, 0.2, 1.8, 0.2])),
            2: EdgeStats(0.5, visits=2, value_sum=np.asarray([1.8, 0.2, 0.2, 0.2])),
        },
        visits=4,
    )
    assert search._select(node)[0] == 1


def test_puct_is_legal_sparse_and_non_mutating() -> None:
    game = SharedGame.new(11, max_rounds=2)
    actor = game.env.whose_turn()
    property_ = game.env.properties[1]
    property_.owner = actor
    game.env.players[actor].properties.append(property_)
    before = random.getstate()
    cash = game.env.players[actor].cash
    model = MonopolyZeroNet()
    model.load_ppo_actor(PPO)
    result = MaxNPUCT(model, SearchConfig(simulations=8, max_depth=16)).choose_action(game, actor, 99)
    assert result.chosen_action in game.env.get_allowed_actions(actor)
    assert sum(result.visits.values()) == 8
    assert all(len(vector) == 4 for vector in result.q_values.values())
    assert sum(result.root_value) == pytest.approx(1.0)
    assert game.env.players[actor].cash == cash
    assert random.getstate() == before


def test_chance_node_materializes_every_ordered_outcome() -> None:
    chance = ChanceNode()
    assert tuple(chance.outcomes) == DICE_OUTCOMES
    assert sum(item.probability for item in chance.outcomes.values()) == pytest.approx(1.0)

