"""
Game State: Player and Property data structures, plus the 240-dim state vector
as described in the paper (Section IV-A).
"""

import numpy as np

from .constants import (
    COLOR_GROUPS,
    MAX_HOUSES,
    NUM_PLAYERS,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    STARTING_CASH,
)


class Property:
    """Represents one purchasable property on the board."""

    def __init__(self, square_id: int):
        self.square_id = square_id
        self.data = PROPERTIES[square_id]
        self.name = self.data["name"]
        self.price = self.data["price"]
        self.mortgage_v = self.data["mortgage"]
        self.color = self.data["color"]
        self.owner = None  # None = bank, 0-3 = player index
        self.mortgaged = False
        self.houses = 0  # 0-4 houses  or  5 = hotel
        self.is_monopoly = False  # True if owner has full color group

    @property
    def is_real_estate(self):
        return self.color not in ("railroad", "utility")

    def calculate_net_worth(self) -> float:
        """Net worth contribution of this property (eq. 3 in paper)."""
        bp = self.price
        mv = self.mortgage_v if self.mortgaged else 0
        b = 5.0 if self.is_monopoly else 2.5
        if self.is_real_estate and self.houses > 0:
            hp = self.data["house_price"]
            # House multiplier grows with development level, reflecting that
            # each additional house generates rent far beyond its build cost.
            # tier 1 house = 1.5×, tier 2 = 2.0×, tier 3 = 2.5×,
            # tier 4 = 3.0×, hotel (5) = 3.5×
            house_multiplier = 1.0 + self.houses * 0.5
            if self.houses == 5:  # hotel
                return (bp - mv) * b + (5 * hp) * house_multiplier
            return (bp - mv) * b + self.houses * hp * house_multiplier
        return (bp - mv) * b

    def get_rent(
        self, dice_roll: int = 7, num_railroads: int = 1, num_utilities: int = 1
    ) -> int:
        """Calculate rent owed when landing on this property."""
        if self.mortgaged or self.owner is None:
            return 0
        if self.color == "railroad":
            idx = min(num_railroads - 1, 3)
            return self.data["rent"][idx]
        if self.color == "utility":
            idx = 0 if num_utilities == 1 else 1
            return self.data["rent"][idx] * dice_roll
        # Real estate
        if self.houses == 0:
            base = self.data["rent"][0]
            return base * 2 if self.is_monopoly else base
        return self.data["rent"][min(self.houses, 5)]

    def __repr__(self):
        return f"Property({self.name}, owner={self.owner}, houses={self.houses})"


class Player:
    """Represents a single Monopoly player."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.cash = STARTING_CASH
        self.position = 0  # board square
        self.in_jail = False
        self.jail_turns = 0
        self.gooj_card = False  # get-out-of-jail-free card
        self.bankrupt = False
        self.properties = []  # list of Property objects owned

    def net_worth(self) -> float:
        """Equation (2) in the paper."""
        return self.cash + sum(p.calculate_net_worth() for p in self.properties)

    def num_monopolies(self) -> int:
        return sum(1 for p in self.properties if p.is_monopoly)

    def railroads_owned(self) -> int:
        return sum(1 for p in self.properties if p.color == "railroad")

    def utilities_owned(self) -> int:
        return sum(1 for p in self.properties if p.color == "utility")

    def can_afford(self, amount: int) -> bool:
        return self.cash >= amount

    def __repr__(self):
        return (
            f"Player({self.player_id}, cash={self.cash}, "
            f"pos={self.position}, nw={self.net_worth():.0f})"
        )


# ── State Vector Construction ──────────────────────────────────────────────────


def build_state_vector(players, properties_dict, agent_id: int) -> np.ndarray:
    """
    Build the 240-dimensional state vector for the learning agent.

    Layout (as in paper Section IV-A):
      - Player representation  : 4 players × 4 features = 16 dims
          [position/40, cash/5000, in_jail, has_gooj_card]
      - Property representation: 28 properties × 8 features = 224 dims
          [owner_onehot(5), mortgaged, is_monopoly, improvement_fraction]

    The agent's own player comes first in the player section.
    """
    state = np.zeros(240, dtype=np.float32)
    idx = 0

    # ── Player features (16 dims) ──
    order = [agent_id] + [i for i in range(NUM_PLAYERS) if i != agent_id]
    for pid in order:
        p = players[pid]
        state[idx] = p.position / 39.0
        state[idx + 1] = min(p.cash / 5000.0, 1.0)
        state[idx + 2] = float(p.in_jail)
        state[idx + 3] = float(p.gooj_card)
        idx += 4

    # ── Property features (224 dims) ──
    for sq in PROPERTY_IDS:
        prop = properties_dict[sq]
        # owner: one-hot of size 5 (bank=all zeros, players 0-3)
        owner_vec = np.zeros(5)
        if prop.owner is not None:
            owner_vec[prop.owner] = 1.0
        state[idx : idx + 5] = owner_vec
        idx += 5
        # mortgaged
        state[idx] = float(prop.mortgaged)
        idx += 1
        # is_monopoly
        state[idx] = float(prop.is_monopoly)
        idx += 1
        # improvement fraction (houses/4 for RE, 0 for others)
        if prop.is_real_estate:
            state[idx] = prop.houses / 5.0  # 5 = hotel
        idx += 1

    assert idx == 240, f"State vector size mismatch: {idx}"
    return state
