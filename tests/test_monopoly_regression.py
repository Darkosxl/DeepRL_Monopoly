from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = ROOT / "RL_PPO(UNOFFICIAL)_MONOPOLY"
sys.path.insert(0, str(PPO_ROOT))

from monopoly_drl.actions import (  # noqa: E402
    ACTION_SPACE_SIZE,
    ActionType,
    AuctionAction,
)
from monopoly_drl.env import (  # noqa: E402
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    MonopolyEnv,
)


class MonopolyRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def test_public_dimensions(self) -> None:
        self.assertEqual(ACTION_SPACE_SIZE, 2958)
        self.assertEqual(self.env._get_state(0).shape, (240,))

    def test_turn_sequence_and_property_purchase(self) -> None:
        self.assertEqual(
            self.env.get_allowed_actions(0), [int(ActionType.END_TURN)]
        )

        self.env.step(int(ActionType.END_TURN))
        self.assertEqual(self.env.phase, PHASE_POST_ROLL)
        self.assertEqual(
            self.env.get_allowed_actions(0), [int(ActionType.ROLL_DICE)]
        )

        with patch("monopoly_drl.env.random.randint", side_effect=[1, 2]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertEqual(self.env.players[0].position, 3)
        self.assertIn(int(ActionType.BUY_PROPERTY), self.env.get_allowed_actions(0))
        self.env.step(int(ActionType.BUY_PROPERTY))
        self.assertEqual(self.env.properties[3].owner, 0)
        self.assertEqual(self.env.players[0].cash, 1440)

        self.env.step(int(ActionType.END_TURN))
        self.assertEqual(self.env.phase, PHASE_OUT_OF_TURN)
        self.assertEqual(self.env.whose_turn(), 1)

    def test_rent_uses_property_table(self) -> None:
        prop = self.env.properties[39]
        prop.owner = 1
        self.env.players[1].properties.append(prop)
        self.env.players[0].position = 37
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        with patch("monopoly_drl.env.random.randint", side_effect=[1, 1]):
            _, _, _, info = self.env.step(int(ActionType.ROLL_DICE))

        self.assertEqual(self.env.players[0].position, 39)
        self.assertEqual(info["rent_paid"], 50)
        self.assertEqual(self.env.players[0].cash, 1450)
        self.assertEqual(self.env.players[1].cash, 1550)

    def test_declined_property_runs_agent_auction(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = True
        self.env.players[0].position = 1

        self.env.step(int(ActionType.END_TURN))
        self.assertEqual(self.env.phase, PHASE_AUCTION)
        self.assertEqual(self.env.whose_turn(), 0)

        self.env.step(int(AuctionAction.BID_100))
        for pid in (1, 2, 3):
            self.assertEqual(self.env.whose_turn(), pid)
            self.env.step(int(AuctionAction.PASS))

        self.assertEqual(self.env.properties[1].owner, 0)
        self.assertEqual(self.env.players[0].cash, 1400)
        self.assertEqual(self.env.phase, PHASE_OUT_OF_TURN)

    def test_three_consecutive_doubles_send_player_to_jail(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        for _ in range(2):
            with patch("monopoly_drl.env.random.randint", side_effect=[1, 1]):
                self.env.step(int(ActionType.ROLL_DICE))
            self.env.step(int(ActionType.END_TURN))
            self.assertEqual(self.env.phase, PHASE_POST_ROLL)
            self.assertFalse(self.env.has_rolled)

        with patch("monopoly_drl.env.random.randint", side_effect=[1, 1]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertTrue(self.env.players[0].in_jail)
        self.assertEqual(self.env.players[0].position, 10)
        self.assertFalse(self.env.extra_roll_pending)

    def test_jail_doubles_release_without_extra_roll(self) -> None:
        player = self.env.players[0]
        player.position = 10
        player.in_jail = True
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        with patch("monopoly_drl.env.random.randint", side_effect=[2, 2]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertFalse(player.in_jail)
        self.assertEqual(player.position, 14)
        self.assertFalse(self.env.extra_roll_pending)


if __name__ == "__main__":
    unittest.main()
