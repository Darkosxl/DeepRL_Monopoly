"""
Action Space (Section IV-B of the paper).

Total: 2922 dimensions
  - Binary actions          :   9  (do_nothing, roll_dice, buy_property,
                                     mortgage/unmortgage property represented
                                     separately, use_gooj_card, end_turn,
                                     declare_bankruptcy)
  - Mortgage / Unmortgage   :  28 + 28 = 56
  - Sell house / Sell hotel :  22 + 22 = 44
  - Improve property        :  22 + 22 = 44  (house / hotel)
  - Sell property to bank   :  28
  - Make buy-trade offer    : 252  (3 players × 28 properties × 3 price levels)
  - Make sell-trade offer   : 252
  - Make exchange offer     : 2268 (3 players × 28 × 27)
  - Accept trade offer      :   1  (binary)
  - Decline trade offer     :   1  (binary)

We index each action with a unique integer and provide mappings.
"""

from .constants import (
    PROPERTY_IDS, REAL_ESTATE_IDS, NUM_PLAYERS, TRADE_CASH_LEVELS
)
from enum import IntEnum


# ── Action enum for non-property actions ──────────────────────────────────────
class ActionType(IntEnum):
    DO_NOTHING       = 0
    END_TURN         = 1
    ROLL_DICE        = 2
    BUY_PROPERTY     = 3   # fixed-policy in hybrid agent
    USE_GOOJ_CARD    = 4
    PAY_BAIL         = 5
    DECLARE_BANKRUPT = 6
    ACCEPT_TRADE     = 7   # fixed-policy in hybrid agent
    DECLINE_TRADE    = 8


NUM_BINARY = len(ActionType)           # 9

# Property-indexed actions
NUM_MORTGAGE   = len(PROPERTY_IDS)     # 28
NUM_UNMORTGAGE = len(PROPERTY_IDS)     # 28
NUM_IMPROVE_H  = len(REAL_ESTATE_IDS)  # 22 (build house)
NUM_IMPROVE_HT = len(REAL_ESTATE_IDS)  # 22 (build hotel)
NUM_SELL_H     = len(REAL_ESTATE_IDS)  # 22
NUM_SELL_HT    = len(REAL_ESTATE_IDS)  # 22
NUM_SELL_PROP  = len(PROPERTY_IDS)     # 28 (sell back to bank at mortgage value)

OTHER_PLAYERS  = NUM_PLAYERS - 1       # 3
NUM_TRADE_CASH = len(TRADE_CASH_LEVELS)  # 3

NUM_BUY_TRADE_OFFER  = OTHER_PLAYERS * len(PROPERTY_IDS) * NUM_TRADE_CASH   # 252
NUM_SELL_TRADE_OFFER = OTHER_PLAYERS * len(PROPERTY_IDS) * NUM_TRADE_CASH   # 252
NUM_EXCH_TRADE_OFFER = OTHER_PLAYERS * len(PROPERTY_IDS) * (len(PROPERTY_IDS) - 1)  # 2268

# Offsets
_o = {}
cur = 0
_o["binary"]           = cur; cur += NUM_BINARY
_o["mortgage"]         = cur; cur += NUM_MORTGAGE
_o["unmortgage"]       = cur; cur += NUM_UNMORTGAGE
_o["improve_house"]    = cur; cur += NUM_IMPROVE_H
_o["improve_hotel"]    = cur; cur += NUM_IMPROVE_HT
_o["sell_house"]       = cur; cur += NUM_SELL_H
_o["sell_hotel"]       = cur; cur += NUM_SELL_HT
_o["sell_prop"]        = cur; cur += NUM_SELL_PROP
_o["buy_trade"]        = cur; cur += NUM_BUY_TRADE_OFFER
_o["sell_trade"]       = cur; cur += NUM_SELL_TRADE_OFFER
_o["exch_trade"]       = cur; cur += NUM_EXCH_TRADE_OFFER

ACTION_SPACE_SIZE = cur   # should be ~2922

OFFSETS = _o


def action_to_description(action_idx: int) -> str:
    """Human-readable action description for debugging."""
    for name, start in sorted(OFFSETS.items(), key=lambda x: x[1]):
        size = _section_size(name)
        if start <= action_idx < start + size:
            local = action_idx - start
            if name == "binary":
                return ActionType(local).name
            if name in ("mortgage", "unmortgage", "sell_prop"):
                prop = PROPERTY_IDS[local % len(PROPERTY_IDS)]
                return f"{name}(sq={prop})"
            if name in ("improve_house", "improve_hotel", "sell_house", "sell_hotel"):
                prop = REAL_ESTATE_IDS[local % len(REAL_ESTATE_IDS)]
                return f"{name}(sq={prop})"
            if name in ("buy_trade", "sell_trade"):
                player_idx = local // (len(PROPERTY_IDS) * NUM_TRADE_CASH)
                rem        = local % (len(PROPERTY_IDS) * NUM_TRADE_CASH)
                prop_idx   = rem // NUM_TRADE_CASH
                price_idx  = rem % NUM_TRADE_CASH
                return (f"{name}(player={player_idx}, "
                        f"prop={PROPERTY_IDS[prop_idx]}, "
                        f"price={TRADE_CASH_LEVELS[price_idx]}x)")
            if name == "exch_trade":
                n_props = len(PROPERTY_IDS)
                player_idx  = local // (n_props * (n_props - 1))
                rem         = local % (n_props * (n_props - 1))
                offer_idx   = rem // (n_props - 1)
                req_idx_raw = rem % (n_props - 1)
                req_idx     = req_idx_raw if req_idx_raw < offer_idx else req_idx_raw + 1
                return (f"exch_trade(player={player_idx}, "
                        f"offer={PROPERTY_IDS[offer_idx]}, "
                        f"req={PROPERTY_IDS[req_idx]})")
            return f"{name}[{local}]"
    return f"UNKNOWN({action_idx})"


def _section_size(name: str) -> int:
    keys = sorted(OFFSETS.keys(), key=lambda k: OFFSETS[k])
    for i, k in enumerate(keys):
        if k == name:
            if i + 1 < len(keys):
                return OFFSETS[keys[i+1]] - OFFSETS[name]
            return ACTION_SPACE_SIZE - OFFSETS[name]
    return 0



