"""Helper functions for loading data files.

NOTE: consider moving this file up into the "configs" directory itself,
to accompany the corresponding data files.
"""

import os

import numpy as np
import pandas as pd

# access data files in a way that works both internally and externally
# look for environment variable used by blaze / bazel internally:
# see: https://bazel.build/concepts/dependencies#data-dependencies
TEST_SRCDIR = os.environ.get("TEST_SRCDIR")
if TEST_SRCDIR:
  REPO_DIRPATH = os.path.join(
      TEST_SRCDIR,
      "google3",
      "third_party",
      "py",
      "smart_buildings",
      "smart_control",
  )
else:
  REPO_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "..")

DIRPATH = os.path.join(REPO_DIRPATH, "configs", "resources", "sb1")

ZONE_TEMPS_FILEPATH = os.path.join(DIRPATH, "reset_temps.npy")
FLOOR_PLAN_FILEPATH = os.path.join(DIRPATH, "double_resolution_zone_1_2.npy")
WEATHER_DATA_DIRPATH = os.path.join(DIRPATH, "weather_data")


def get_floor_plan(filepath: str = FLOOR_PLAN_FILEPATH) -> np.ndarray:
  """Returns the floor plan as a numpy array."""
  with open(filepath, "rb") as f:
    return np.load(f)


def get_zone_temps(filepath: str = ZONE_TEMPS_FILEPATH) -> np.ndarray:
  """Returns the zone temperatures as a numpy array."""
  with open(filepath, "rb") as f:
    return np.load(f)


def get_weather_data_filepath(year: int = 2024) -> str:
  """Returns the filepath to the weather data for the given year."""
  return os.path.join(WEATHER_DATA_DIRPATH, f"{year}.csv")


def get_weather_data(year: int = 2024) -> pd.DataFrame:
  """Returns the weather data for the given year as a pandas DataFrame."""
  return pd.read_csv(get_weather_data_filepath(year))
