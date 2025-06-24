"""Reinforcement learning constants."""

import os

from smart_control.utils.constants import ROOT_DIR

# Relative filepaths:
RL_DIR = os.path.join(ROOT_DIR, 'smart_control', 'reinforcement_learning')
RL_EXPERIMENT_RESULTS_DIR = os.path.join(RL_DIR, 'experiment_results')
RL_EXPERIMENT_METRICS_DIR = os.path.join(RL_EXPERIMENT_RESULTS_DIR, 'metrics')
RL_EXPERIMENT_RENDERS_DIR = os.path.join(RL_EXPERIMENT_RESULTS_DIR, 'renders')
# RL_STARTER_BUFFERS_DIR = os.path.join(RL_DIR, 'data', 'starter_buffers')

# Default time zone for plotting and simulations
DEFAULT_TIME_ZONE = 'US/Pacific'

# Economic constants
PERSON_PRODUCTIVITY_HOUR = 300.0

# Reward adjustments
REWARD_SHIFT = 0
REWARD_SCALE = 1.0


DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT = 125.0
