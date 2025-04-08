# config.py

import math

# ---------------------
# Problem Definition and Environment Settings
# ---------------------
OPERATION_START_MIN = 6 * 60       # 6:00 AM = 360 minutes
OPERATION_END_MIN   = 21 * 60      # 9:00 PM = 1260 minutes
T_RANGE = OPERATION_END_MIN - OPERATION_START_MIN  # total operating period

# Bus line definitions:
# Two bus lines with loop service:
# Bus Line 1: Terminal1 (trip: 25 min + rest: 5 min; interval = 30 minutes)
# Bus Line 2: Terminal3 (trip: 40 min + rest: 5 min; interval = 45 minutes)
BUS_LINES = {
    1: {
        "name": "Bus Line 1",
        "terminal": "Terminal1",  
        "interval": 30,
        "trip_time": 25,
        "rest_time": 5
    },
    2: {
        "name": "Bus Line 2",
        "terminal": "Terminal3",  
        "interval": 45,
        "trip_time": 40,
        "rest_time": 5
    }
}

DEPOT = "Depot"
INITIAL_NUM_BUSES = 10  # total buses available at the depot

# Coordinates for locations (for deadhead cost calculation)
COORDINATES = {
    DEPOT: (0, 0),
    "Terminal1": (1, 0),
    "Terminal3": (0, 1)
}
MAX_DISTANCE = math.sqrt((1-0)**2 + (1-0)**2)  # ≈1.414

# ---------------------
# RL and PPO Hyperparameters
# ---------------------
# New state vector: 2 base features (normalized current time, normalized bus line) +
# a continuous availability value for each bus.
STATE_DIM = 2 + INITIAL_NUM_BUSES  
ACTION_DIM = INITIAL_NUM_BUSES  
MAX_EPISODE_STEPS = None  # set dynamically based on timetable length

HIDDEN_SIZE = 64
LEARNING_RATE = 1e-5
GAMMA = 0.99
CLIP_EPS = 0.1
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
NUM_EPISODES = 30000  # increased training episodes

# ---------------------
# Reward Weights (tuning for chaining and fewer buses)
# ---------------------
W_UNUSED_PENALTY = 20.0      # penalty for selecting an unused bus when a used bus is available
W_DEADHEAD = 5.0             # weight for deadhead cost (normalized)
W_UNAVAILABILITY = 20.0      # penalty if bus isn’t ready (if available value < 0)
W_REST_REWARD = 10.0         # reward for reusing a bus that is already active
W_DEMAND_PENALTY = 1.0       # (kept as is)
W_CHAIN = 5.0              # additional bonus if the chosen bus was used in the previous trip on the same bus line
W_FINAL = 50.0             # final penalty per bus used
