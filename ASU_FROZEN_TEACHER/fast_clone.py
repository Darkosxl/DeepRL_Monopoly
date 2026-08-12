"""Hand-written ``MonopolyEnv`` clone that avoids ``copy.deepcopy``'s overhead.

``copy.deepcopy`` pays for generic memo-dict bookkeeping and recursive
introspection on every call. This module knows the environment's actual
mutable structure and rebuilds only that, sharing every field that is never
reassigned after construction. Nothing here changes behavior versus
``copy.deepcopy(env)`` -- see ``tests/test_asu_rollout_v2.py`` for the
equivalence tests that back this claim.
"""

from __future__ import annotations

import random

from monopoly_game_engine.env import MonopolyEnv, TradeOffer
from monopoly_game_engine.state import Player, Property

from .core import _PrivateGame


def _clone_property(prop: Property) -> Property:
    clone = Property.__new__(Property)
    # Construction-time-only fields: never reassigned after __init__
    # anywhere in monopoly_game_engine (verified by grep), safe to share.
    clone.square_id = prop.square_id
    clone.data = prop.data
    clone.name = prop.name
    clone.price = prop.price
    clone.mortgage_v = prop.mortgage_v
    clone.color = prop.color
    # Mutable fields: copy the value fresh per clone.
    clone.owner = prop.owner
    clone.mortgaged = prop.mortgaged
    clone.houses = prop.houses
    clone.is_monopoly = prop.is_monopoly
    return clone


def _clone_player(player: Player, prop_clone: dict[int, Property]) -> Player:
    clone = Player.__new__(Player)
    clone.player_id = player.player_id
    clone.cash = player.cash
    clone.position = player.position
    clone.in_jail = player.in_jail
    clone.jail_turns = player.jail_turns
    clone.gooj_card = player.gooj_card
    clone.bankrupt = player.bankrupt
    # Same Property objects as the cloned env.properties dict -- required
    # since monopoly_game_engine has no __eq__/__hash__ anywhere (verified
    # by grep) and relies on default identity equality for list.remove()/
    # `in` checks on this list.
    clone.properties = [prop_clone[id(prop)] for prop in player.properties]
    return clone


_ENV_SCALAR_FIELDS = (
    "current_turn_idx",
    "round",
    "done",
    "last_dice",
    "phase",
    "has_rolled",
    "consecutive_doubles",
    "extra_roll_pending",
    "auction_property_id",
    "auction_current_pid",
    "auction_high_bid",
    "auction_high_bidder",
    "houses_available",
    "hotels_available",
    "debt_player",
    "debt_creditor",
    "debt_amount",
    "player_needs_funds",
)


def fast_clone_env(env: MonopolyEnv) -> MonopolyEnv:
    """Behaviorally-equivalent, faster replacement for ``copy.deepcopy(env)``.

    Every ``MonopolyEnv``/``Property``/``Player``/``TradeOffer`` field is
    accounted for (cross-checked against ``MonopolyEnv.__init__``/``reset()``
    and every ``self.<attr> =`` assignment in ``monopoly_game_engine/env.py``)
    -- see the module docstring for why this is safe.
    """

    prop_clone: dict[int, Property] = {}
    new_properties: dict[int, Property] = {}
    for square, prop in env.properties.items():
        clone = _clone_property(prop)
        new_properties[square] = clone
        prop_clone[id(prop)] = clone

    new_players = [_clone_player(player, prop_clone) for player in env.players]

    new_pending_trades = {
        sender: TradeOffer(
            offer.from_player,
            offer.to_player,
            offered_prop=(
                prop_clone[id(offer.offered_prop)]
                if offer.offered_prop is not None
                else None
            ),
            requested_prop=(
                prop_clone[id(offer.requested_prop)]
                if offer.requested_prop is not None
                else None
            ),
            cash_offered=offer.cash_offered,
            cash_requested=offer.cash_requested,
        )
        for sender, offer in env.pending_trades.items()
    }

    clone = MonopolyEnv.__new__(MonopolyEnv)
    clone.agent_ids = env.agent_ids
    clone.max_rounds = env.max_rounds
    clone.players = new_players
    clone.properties = new_properties
    clone.pending_trades = new_pending_trades
    clone.turn_order = list(env.turn_order)
    clone.out_of_turn_pids = list(env.out_of_turn_pids)
    clone.auction_bidders = list(env.auction_bidders)
    for field in _ENV_SCALAR_FIELDS:
        setattr(clone, field, getattr(env, field))
    return clone


class _FastPrivateGame(_PrivateGame):
    """``_PrivateGame`` (``ASU_FROZEN_TEACHER.core``) using the fast clone."""

    __slots__ = ()

    def __init__(self, env: MonopolyEnv, seed: int):
        self.env = fast_clone_env(env)
        self.random_state = random.Random(seed).getstate()


__all__ = ["fast_clone_env", "_FastPrivateGame"]
