"""
Monopoly Constants - Board layout, property prices, color groups, etc.
Based on the US standard Monopoly board.
"""

# ── Board Squares ──────────────────────────────────────────────────────────────
# Square index → name
BOARD = {
    0:  "Go",
    1:  "Mediterranean Avenue",
    2:  "Community Chest",
    3:  "Baltic Avenue",
    4:  "Income Tax",
    5:  "Reading Railroad",
    6:  "Oriental Avenue",
    7:  "Chance",
    8:  "Vermont Avenue",
    9:  "Connecticut Avenue",
    10: "Jail / Just Visiting",
    11: "St. Charles Place",
    12: "Electric Company",
    13: "States Avenue",
    14: "Virginia Avenue",
    15: "Pennsylvania Railroad",
    16: "St. James Place",
    17: "Community Chest",
    18: "Tennessee Avenue",
    19: "New York Avenue",
    20: "Free Parking",
    21: "Kentucky Avenue",
    22: "Chance",
    23: "Indiana Avenue",
    24: "Illinois Avenue",
    25: "B&O Railroad",
    26: "Atlantic Avenue",
    27: "Ventnor Avenue",
    28: "Water Works",
    29: "Marvin Gardens",
    30: "Go to Jail",
    31: "Pacific Avenue",
    32: "North Carolina Avenue",
    33: "Community Chest",
    34: "Pennsylvania Avenue",
    35: "Short Line Railroad",
    36: "Chance",
    37: "Park Place",
    38: "Luxury Tax",
    39: "Boardwalk",
}

# ── Property Definitions ────────────────────────────────────────────────────────
# Each property: (board_index, price, mortgage_value, color_group, rent_levels)
# rent_levels = [base, 1h, 2h, 3h, 4h, hotel]
# Railroads: [base, x2, x3, x4]
# Utilities: multipliers

PROPERTIES = {
    # Brown / Purple
    1:  {"name": "Mediterranean Avenue", "price": 60,  "mortgage": 30,  "color": "brown",    "house_price": 50,  "rent": [2, 10, 30, 90, 160, 250]},
    3:  {"name": "Baltic Avenue",        "price": 60,  "mortgage": 30,  "color": "brown",    "house_price": 50,  "rent": [4, 20, 60, 180, 320, 450]},
    # Light Blue
    6:  {"name": "Oriental Avenue",      "price": 100, "mortgage": 50,  "color": "lightblue","house_price": 50,  "rent": [6, 30, 90, 270, 400, 550]},
    8:  {"name": "Vermont Avenue",       "price": 100, "mortgage": 50,  "color": "lightblue","house_price": 50,  "rent": [6, 30, 90, 270, 400, 550]},
    9:  {"name": "Connecticut Avenue",   "price": 120, "mortgage": 60,  "color": "lightblue","house_price": 50,  "rent": [8, 40, 100, 300, 450, 600]},
    # Pink
    11: {"name": "St. Charles Place",    "price": 140, "mortgage": 70,  "color": "pink",     "house_price": 100, "rent": [10, 50, 150, 450, 625, 750]},
    13: {"name": "States Avenue",        "price": 140, "mortgage": 70,  "color": "pink",     "house_price": 100, "rent": [10, 50, 150, 450, 625, 750]},
    14: {"name": "Virginia Avenue",      "price": 160, "mortgage": 80,  "color": "pink",     "house_price": 100, "rent": [12, 60, 180, 500, 700, 900]},
    # Orange
    16: {"name": "St. James Place",      "price": 180, "mortgage": 90,  "color": "orange",   "house_price": 100, "rent": [14, 70, 200, 550, 750, 950]},
    18: {"name": "Tennessee Avenue",     "price": 180, "mortgage": 90,  "color": "orange",   "house_price": 100, "rent": [14, 70, 200, 550, 750, 950]},
    19: {"name": "New York Avenue",      "price": 200, "mortgage": 100, "color": "orange",   "house_price": 100, "rent": [16, 80, 220, 600, 800, 1000]},
    # Red
    21: {"name": "Kentucky Avenue",      "price": 220, "mortgage": 110, "color": "red",      "house_price": 150, "rent": [18, 90, 250, 700, 875, 1050]},
    23: {"name": "Indiana Avenue",       "price": 220, "mortgage": 110, "color": "red",      "house_price": 150, "rent": [18, 90, 250, 700, 875, 1050]},
    24: {"name": "Illinois Avenue",      "price": 240, "mortgage": 120, "color": "red",      "house_price": 150, "rent": [20, 100, 300, 750, 925, 1100]},
    # Yellow
    26: {"name": "Atlantic Avenue",      "price": 260, "mortgage": 130, "color": "yellow",   "house_price": 150, "rent": [22, 110, 330, 800, 975, 1150]},
    27: {"name": "Ventnor Avenue",       "price": 260, "mortgage": 130, "color": "yellow",   "house_price": 150, "rent": [22, 110, 330, 800, 975, 1150]},
    29: {"name": "Marvin Gardens",       "price": 280, "mortgage": 140, "color": "yellow",   "house_price": 150, "rent": [24, 120, 360, 850, 1025, 1200]},
    # Green
    31: {"name": "Pacific Avenue",       "price": 300, "mortgage": 150, "color": "green",    "house_price": 200, "rent": [26, 130, 390, 900, 1100, 1275]},
    32: {"name": "North Carolina Ave",   "price": 300, "mortgage": 150, "color": "green",    "house_price": 200, "rent": [26, 130, 390, 900, 1100, 1275]},
    34: {"name": "Pennsylvania Avenue",  "price": 320, "mortgage": 160, "color": "green",    "house_price": 200, "rent": [28, 150, 450, 1000, 1200, 1400]},
    # Dark Blue
    37: {"name": "Park Place",           "price": 350, "mortgage": 175, "color": "darkblue", "house_price": 200, "rent": [35, 175, 500, 1100, 1300, 1500]},
    39: {"name": "Boardwalk",            "price": 400, "mortgage": 200, "color": "darkblue", "house_price": 200, "rent": [50, 200, 600, 1400, 1700, 2000]},
    # Railroads
    5:  {"name": "Reading Railroad",     "price": 200, "mortgage": 100, "color": "railroad", "rent": [25, 50, 100, 200]},
    15: {"name": "Pennsylvania Railroad","price": 200, "mortgage": 100, "color": "railroad", "rent": [25, 50, 100, 200]},
    25: {"name": "B&O Railroad",         "price": 200, "mortgage": 100, "color": "railroad", "rent": [25, 50, 100, 200]},
    35: {"name": "Short Line Railroad",  "price": 200, "mortgage": 100, "color": "railroad", "rent": [25, 50, 100, 200]},
    # Utilities
    12: {"name": "Electric Company",     "price": 150, "mortgage": 75,  "color": "utility",  "rent": [4, 10]},  # multipliers of dice roll
    28: {"name": "Water Works",          "price": 150, "mortgage": 75,  "color": "utility",  "rent": [4, 10]},
}

PROPERTY_IDS = sorted(PROPERTIES.keys())   # list of 28 property squares
REAL_ESTATE_IDS = [p for p in PROPERTY_IDS if PROPERTIES[p]["color"] not in ("railroad", "utility")]
RAILROAD_IDS    = [p for p in PROPERTY_IDS if PROPERTIES[p]["color"] == "railroad"]
UTILITY_IDS     = [p for p in PROPERTY_IDS if PROPERTIES[p]["color"] == "utility"]    

# Color groups: color → list of property indices
COLOR_GROUPS = {}
for pid, pdata in PROPERTIES.items():
    COLOR_GROUPS.setdefault(pdata["color"], []).append(pid)

# Tax squares
INCOME_TAX_SQUARE  = 4   # pay $200
LUXURY_TAX_SQUARE  = 38  # pay $100
JAIL_SQUARE        = 10
GO_TO_JAIL_SQUARE  = 30
GO_SQUARE          = 0
FREE_PARKING       = 20
GO_SALARY          = 200

# Starting cash
STARTING_CASH = 1500

# Max houses per property
MAX_HOUSES = 4
MAX_HOTELS = 1

# Jail rules
MAX_JAIL_TURNS = 3
JAIL_BAIL      = 50

# Trade cash levels (fraction of purchase price)
TRADE_CASH_LEVELS = [0.75, 1.0, 1.25] # Work To Do Here

NUM_PLAYERS = 4 # Work To Do Here


CHANCE_CARDS = [
    "Advance to Go (Collect $200)",
    "Advance to Illinois Ave.",
    "Advance to St. Charles Place",
    "Advance token to nearest Railroad",
    "Advance token to nearest Utility",
    "Bank pays you dividend of $50",
    "Get Out of Jail Free",
    "Go Back 3 Spaces",
    "Go to Jail. Go directly to Jail.",
    "Make general repairs on all your property – $25 per house, $100 per hotel",
    "Pay poor tax of $15",
    "Take a trip to Reading Railroad",
    "Take a walk on the Boardwalk",
    "You have been elected Chairman of the Board – Pay each player $50",
    "Your building loan matures – Collect $150",
    "You have won a crossword competition – Collect $100",
]

COMMUNITY_CHEST_CARDS = [
    "Advance to Go (Collect $200)",
    "Bank error in your favor – Collect $200",
    "Doctor's fees – Pay $50",
    "From sale of stock you get $50",
    "Get Out of Jail Free",
    "Go to Jail. Go directly to Jail.",
    "Grand Opera Night – Collect $50 from every player",
    "Holiday Fund matures – Receive $100",
    "Income tax refund – Collect $20",
    "It is your birthday – Collect $10 from every player",
    "Life insurance matures – Collect $100",
    "Pay hospital fees of $100",
    "Pay school fees of $150",
    "Receive $25 consultancy fee",
    "You are assessed for street repairs – $40 per house, $115 per hotel",
    "You have won second prize in a beauty contest – Collect $10",
    "You inherit $100",
]

CHANCE_SQUARES     = {7, 22, 36}
COMMUNITY_SQUARES  = {2, 17, 33}