from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from monopoly_game_engine.actions import (  # noqa: E402
    ACTION_SPACE_SIZE,
    OFFSETS,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.agents_fixed import TheGambler, TheHoarder  # noqa: E402
from monopoly_game_engine.env import (  # noqa: E402
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    MonopolyEnv,
    TradeOffer,
)


class MonopolyRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def give_property(self, square: int, pid: int, mortgaged: bool = False) -> None:
        prop = self.env.properties[square]
        prop.owner = pid
        prop.mortgaged = mortgaged
        self.env.players[pid].properties.append(prop)
        self.env._update_monopolies()

    def test_public_dimensions(self) -> None:
        self.assertEqual(ACTION_SPACE_SIZE, 2958)
        state = self.env._get_state(0)
        self.assertEqual(state.shape, (300,))
        self.assertEqual(float(state[240:244].sum()), 1.0)
        self.assertEqual(float(state[244:248].sum()), 1.0)
        self.assertEqual(float(state[248:252].sum()), 1.0)

    def test_observation_distinguishes_public_pending_state(self) -> None:
        baseline = self.env._get_state(0)

        self.env.pending_trades[2] = TradeOffer(2, 0, cash_offered=100)
        with_trade = self.env._get_state(0)
        self.assertFalse((baseline == with_trade).all())

        self.env.pending_trades[2] = TradeOffer(2, 0, cash_offered=200)
        different_terms = self.env._get_state(0)
        self.assertFalse((with_trade == different_terms).all())

        self.env.pending_trades = {}
        self.env.turn_order = [0, 2, 1, 3]
        reordered = self.env._get_state(0)
        self.assertFalse((baseline == reordered).all())

        self.env.phase = PHASE_AUCTION
        self.env.auction_property_id = 1
        self.env.auction_current_pid = 0
        self.env.auction_bidders = [0, 1]
        two_bidders = self.env._get_state(0)
        self.env.auction_bidders = [0, 1, 2]
        three_bidders = self.env._get_state(0)
        self.assertFalse((two_bidders == three_bidders).all())

    def test_incoming_trade_priority_follows_relative_turn_order(self) -> None:
        self.env.pending_trades[1] = TradeOffer(1, 0, cash_offered=100)
        self.env.pending_trades[3] = TradeOffer(3, 0, cash_offered=300)
        self.assertEqual(self.env._incoming_trade(0).cash_offered, 100)
        self.env._do_accept_trade(0)
        self.assertEqual(self.env.players[0].cash, 1600)
        self.assertNotIn(1, self.env.pending_trades)
        self.assertIn(3, self.env.pending_trades)

        rotated = MonopolyEnv(agent_ids=[2], max_rounds=5)
        rotated.turn_order = [2, 3, 0, 1]
        rotated.pending_trades[3] = TradeOffer(3, 2, cash_offered=100)
        rotated.pending_trades[1] = TradeOffer(1, 2, cash_offered=300)
        self.assertEqual(rotated._incoming_trade(2).cash_offered, 100)
        rotated.step(int(ActionType.DECLINE_TRADE))
        self.assertNotIn(3, rotated.pending_trades)
        self.assertIn(1, rotated.pending_trades)

    def test_turn_sequence_and_property_purchase(self) -> None:
        self.assertEqual(
            self.env.get_allowed_actions(0), [int(ActionType.END_TURN)]
        )

        self.env.step(int(ActionType.END_TURN))
        self.assertEqual(self.env.phase, PHASE_POST_ROLL)
        self.assertEqual(
            self.env.get_allowed_actions(0), [int(ActionType.ROLL_DICE)]
        )

        with patch("monopoly_game_engine.env.random.randint", side_effect=[1, 2]):
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

        with patch("monopoly_game_engine.env.random.randint", side_effect=[1, 1]):
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

    def test_fixed_agents_use_personality_during_auction(self) -> None:
        self.env._start_auction(1)

        self.assertEqual(
            TheHoarder(0).choose_action(self.env), int(AuctionAction.PASS)
        )
        self.assertEqual(
            TheGambler(0).choose_action(self.env), int(AuctionAction.BID_50)
        )

    def test_three_consecutive_doubles_send_player_to_jail(self) -> None:
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        for _ in range(2):
            with patch(
                "monopoly_game_engine.env.random.randint", side_effect=[1, 1]
            ):
                self.env.step(int(ActionType.ROLL_DICE))
            self.env.step(int(ActionType.END_TURN))
            self.assertEqual(self.env.phase, PHASE_POST_ROLL)
            self.assertFalse(self.env.has_rolled)

        with patch("monopoly_game_engine.env.random.randint", side_effect=[1, 1]):
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

        with patch("monopoly_game_engine.env.random.randint", side_effect=[2, 2]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertFalse(player.in_jail)
        self.assertEqual(player.position, 14)
        self.assertFalse(self.env.extra_roll_pending)

    def test_liquidation_pays_unpaid_rent(self) -> None:
        self.give_property(39, 1)
        self.give_property(1, 0)
        player = self.env.players[0]
        player.cash = 20
        player.position = 37
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        with patch("monopoly_game_engine.env.random.randint", side_effect=[1, 1]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertEqual(self.env.debt_amount, 30)
        mortgage = OFFSETS["mortgage"] + PROPERTY_IDS.index(1)
        self.assertIn(mortgage, self.env.get_allowed_actions(0))
        with self.assertRaisesRegex(ValueError, "Illegal action"):
            self.env.step(int(ActionType.END_TURN))
        self.env.step(mortgage)

        self.assertIsNone(self.env.debt_player)
        self.assertEqual(player.cash, 0)
        self.assertEqual(self.env.players[1].cash, 1550)

    def test_bankruptcy_transfers_deeds_to_creditor(self) -> None:
        self.give_property(39, 1)
        self.give_property(1, 0, mortgaged=True)
        player = self.env.players[0]
        player.cash = 0
        player.position = 37
        self.env.phase = PHASE_POST_ROLL
        self.env.has_rolled = False

        with patch("monopoly_game_engine.env.random.randint", side_effect=[1, 1]):
            self.env.step(int(ActionType.ROLL_DICE))

        self.assertEqual(
            self.env.get_allowed_actions(0), [int(ActionType.DECLARE_BANKRUPT)]
        )
        self.env.step(int(ActionType.DECLARE_BANKRUPT))

        self.assertTrue(player.bankrupt)
        self.assertEqual(self.env.properties[1].owner, 1)
        self.assertIn(self.env.properties[1], self.env.players[1].properties)
        self.assertEqual(self.env.active_player_id(), 1)
        self.assertEqual(self.env.whose_turn(), 1)

    def test_house_and_hotel_bank_inventory_is_conserved(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 0)
        prop = self.env.properties[1]
        house_action = OFFSETS["improve_house"] + REAL_ESTATE_IDS.index(1)
        hotel_action = OFFSETS["improve_hotel"] + REAL_ESTATE_IDS.index(1)
        sell_hotel_action = OFFSETS["sell_hotel"] + REAL_ESTATE_IDS.index(1)

        self.env.houses_available = 1
        self.env.step(house_action)
        self.assertEqual(prop.houses, 1)
        self.assertEqual(self.env.houses_available, 0)
        self.assertNotIn(house_action, self.env.get_allowed_actions(0))

        prop.houses = 4
        self.env.houses_available = 28
        self.env.step(hotel_action)
        self.assertEqual(prop.houses, 5)
        self.assertEqual(self.env.houses_available, 32)
        self.assertEqual(self.env.hotels_available, 11)

        self.env.step(sell_hotel_action)
        self.assertEqual(prop.houses, 4)
        self.assertEqual(self.env.houses_available, 28)
        self.assertEqual(self.env.hotels_available, 12)

    def test_trade_target_indices_survive_bankruptcy(self) -> None:
        self.env.players[1].bankrupt = True
        self.give_property(1, 2)
        target_slot = [1, 2, 3].index(2)
        buy_offer = (
            OFFSETS["buy_trade"]
            + target_slot * len(PROPERTY_IDS) * 3
            + PROPERTY_IDS.index(1) * 3
        )

        self.assertIn(buy_offer, self.env.get_allowed_actions(0))
        self.env.step(buy_offer)

        self.assertEqual(self.env.pending_trades[0].to_player, 2)
        self.assertIs(self.env.pending_trades[0].requested_prop, self.env.properties[1])

    def test_unaffordable_trade_is_rejected_atomically(self) -> None:
        self.give_property(1, 0)
        sender = self.env.players[0]
        recipient = self.env.players[1]
        recipient.cash = 50
        self.env.pending_trades[0] = TradeOffer(
            0,
            1,
            offered_prop=self.env.properties[1],
            cash_requested=100,
        )

        self.env._do_accept_trade(1)

        self.assertEqual(sender.cash, 1500)
        self.assertEqual(recipient.cash, 50)
        self.assertEqual(self.env.properties[1].owner, 0)
        self.assertIn(self.env.properties[1], sender.properties)
        self.assertNotIn(self.env.properties[1], recipient.properties)

    def test_sell_property_action_is_legal_only_when_unmortgaged(self) -> None:
        self.give_property(1, 0)
        prop = self.env.properties[1]
        sell = OFFSETS["sell_prop"] + PROPERTY_IDS.index(1)

        self.assertIn(sell, self.env.get_allowed_actions(0))
        self.env.step(sell)

        self.assertIsNone(prop.owner)
        self.assertEqual(self.env.players[0].cash, 1530)

        self.give_property(1, 0, mortgaged=True)
        self.assertNotIn(sell, self.env.get_allowed_actions(0))

    def test_mortgaged_deed_cannot_be_improved(self) -> None:
        self.give_property(1, 0, mortgaged=True)
        self.give_property(3, 0)
        prop = self.env.properties[1]
        improve = OFFSETS["improve_house"] + REAL_ESTATE_IDS.index(1)

        self.assertNotIn(improve, self.env.get_allowed_actions(0))
        with self.assertRaisesRegex(ValueError, "Illegal action"):
            self.env.step(improve)

        self.assertEqual(prop.houses, 0)
        self.assertEqual(self.env.players[0].cash, 1500)

    def test_exchange_offer_is_enumerated_after_bankruptcy(self) -> None:
        self.env.players[1].bankrupt = True
        self.give_property(1, 0)
        self.give_property(3, 2)
        target_slot = [1, 2, 3].index(2)
        offer_idx = PROPERTY_IDS.index(1)
        request_idx = PROPERTY_IDS.index(3)
        request_raw = request_idx if request_idx < offer_idx else request_idx - 1
        exchange = (
            OFFSETS["exch_trade"]
            + target_slot * len(PROPERTY_IDS) * (len(PROPERTY_IDS) - 1)
            + offer_idx * (len(PROPERTY_IDS) - 1)
            + request_raw
        )

        self.assertIn(exchange, self.env.get_allowed_actions(0))
        self.env.step(exchange)

        offer = self.env.pending_trades[0]
        self.assertEqual(offer.to_player, 2)
        self.assertIs(offer.offered_prop, self.env.properties[1])
        self.assertIs(offer.requested_prop, self.env.properties[3])

    def test_out_of_turn_offers_only_target_future_responders(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 1)
        self.give_property(5, 3)
        self.env.phase = PHASE_OUT_OF_TURN
        self.env.out_of_turn_pids = [2, 3]

        allowed = self.env.get_allowed_actions(2)
        all_others = [0, 1, 3]
        expired_target = (
            OFFSETS["buy_trade"]
            + all_others.index(1) * len(PROPERTY_IDS) * 3
            + PROPERTY_IDS.index(3) * 3
        )
        future_target = (
            OFFSETS["buy_trade"]
            + all_others.index(3) * len(PROPERTY_IDS) * 3
            + PROPERTY_IDS.index(5) * 3
        )

        self.assertNotIn(expired_target, allowed)
        self.assertIn(future_target, allowed)


if __name__ == "__main__":
    unittest.main()
