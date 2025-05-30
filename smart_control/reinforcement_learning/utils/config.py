<<<<<<< HEAD
=======
"""Reinforcement learning configurations."""

>>>>>>> copybara_push
import os
from typing import Any

import gin
import numpy as np

<<<<<<< HEAD
# the following imports are necessary for proper gin setup, even if they aren't referenced
# do not remove
from smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
from smart_control.reward.setpoint_energy_carbon_regret import (
    SetpointEnergyCarbonRegretFunction,
)
=======
# pylint: disable=unused-import
# these imports are necessary for proper gin setup, even if not referenced
# do not remove
from smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
from smart_control.reward.setpoint_energy_carbon_regret import SetpointEnergyCarbonRegretFunction
>>>>>>> copybara_push
from smart_control.simulator.air_handler import AirHandler
from smart_control.simulator.boiler import Boiler
from smart_control.simulator.building import MaterialProperties
from smart_control.simulator.hvac_floorplan_based import FloorPlanBasedHvac
<<<<<<< HEAD
from smart_control.simulator.randomized_arrival_departure_occupancy import (
    RandomizedArrivalDepartureOccupancy,
)
from smart_control.simulator.simulator_building import SimulatorBuilding
from smart_control.simulator.stochastic_convection_simulator import (
    StochasticConvectionSimulator,
)
from smart_control.simulator.tf_simulator import TFSimulator
from smart_control.simulator.weather_controller import ReplayWeatherController
from smart_control.utils import controller_reader, histogram_reducer
from smart_control.utils.controller_writer import ProtoWriterFactory
from smart_control.utils.environment_utils import to_timestamp
from smart_control.utils.observation_normalizer import (
    StandardScoreObservationNormalizer,
)

ROOT_DIR = "/home/gabriel-user/projects/sbsim/" # Path to the root directory of the project


=======
from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.simulator_building import SimulatorBuilding
from smart_control.simulator.stochastic_convection_simulator import StochasticConvectionSimulator
from smart_control.simulator.tf_simulator import TFSimulator
from smart_control.simulator.weather_controller import ReplayWeatherController
from smart_control.utils import controller_reader
from smart_control.utils import histogram_reducer
from smart_control.utils.constants import ROOT_DIR
from smart_control.utils.controller_writer import ProtoWriterFactory
from smart_control.utils.environment_utils import to_timestamp
from smart_control.utils.observation_normalizer import StandardScoreObservationNormalizer

# pylint: enable=unused-import

# Relative filepaths. Consider moving to reinforcement_learning/constants.py
# pylint: disable=line-too-long
>>>>>>> copybara_push
DATA_PATH = os.path.join(ROOT_DIR, "smart_control", "configs", "resources", "sb1")
CONFIG_PATH = os.path.join(ROOT_DIR, "smart_control", "configs", "resources", "sb1", "train_sim_configs")
METRICS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results", "metrics")
RENDERS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results", "renders")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "data", "starter_buffers")
EXPERIMENT_RESULTS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results")
<<<<<<< HEAD
=======
# pylint: enable=line-too-long
>>>>>>> copybara_push


@gin.configurable
def get_histogram_path() -> str:
<<<<<<< HEAD
    """Get path to histogram data.
    
    Returns:
        Path to histogram data.
    """
    return DATA_PATH
=======
  """Get path to histogram data.

  Returns:
      Path to histogram data.
  """
  return DATA_PATH
>>>>>>> copybara_push


@gin.configurable
def get_reset_temp_values() -> np.ndarray:
<<<<<<< HEAD
    """Get reset temperature values.
    
    Returns:
        Reset temperature values.
    """
    reset_temps_filepath = os.path.join(DATA_PATH, "reset_temps.npy")
    
    return np.load(reset_temps_filepath)
=======
  """Get reset temperature values.

  Returns:
      Reset temperature values.
  """
  reset_temps_filepath = os.path.join(DATA_PATH, "reset_temps.npy")

  return np.load(reset_temps_filepath)
>>>>>>> copybara_push


@gin.configurable
def get_zone_path() -> str:
<<<<<<< HEAD
    """Get path to zone data.
    
    Returns:
        Path to zone data.
    """
    return os.path.join(DATA_PATH, "double_resolution_zone_1_2.npy")
=======
  """Get path to zone data.

  Returns:
      Path to zone data.
  """
  return os.path.join(DATA_PATH, "double_resolution_zone_1_2.npy")
>>>>>>> copybara_push


@gin.configurable
def get_metrics_path() -> str:
<<<<<<< HEAD
    """Get path to metrics.
    
    Returns:
        Path to metrics.
    """
    return os.path.join(METRICS_PATH, "metrics")
=======
  """Get path to metrics.

  Returns:
      Path to metrics.
  """
  return os.path.join(METRICS_PATH, "metrics")
>>>>>>> copybara_push


@gin.configurable
def get_weather_path() -> str:
<<<<<<< HEAD
    """Get path to weather data.
    
    Returns:
        Path to weather data.
    """
    return os.path.join(DATA_PATH, "local_weather_moffett_field_20230701_20231122.csv")
=======
  """Get path to weather data.

  Returns:
      Path to weather data.
  """
  return os.path.join(
      DATA_PATH, "local_weather_moffett_field_20230701_20231122.csv"
  )
>>>>>>> copybara_push


@gin.configurable
def get_histogram_reducer() -> Any:
<<<<<<< HEAD
    """Get histogram reducer.
    
    Returns:
        Histogram reducer.
    """
    
    histogram_parameters_tuples = (
        ('zone_air_temperature_sensor', (285., 286., 287., 288, 289., 290., 291.,
        292., 293., 294., 295., 296., 297., 298., 299., 300., 301, 302, 303)),
        ('supply_air_damper_percentage_command', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ('supply_air_flowrate_setpoint', (0., 0.05, .1, .2, .3, .4, .5,  .7,  .9)),
    )
    
    reader = controller_reader.ProtoReader(DATA_PATH)
    
    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=histogram_parameters_tuples,
        reader=reader,
        normalize_reduce=True,
    )
    return hr
=======
  """Get histogram reducer.

  Returns:
      Histogram reducer.
  """
  # pylint: disable=bad-continuation
  histogram_parameters_tuples = (
      ("zone_air_temperature_sensor", (
          285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0,
          294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0,
      )),
      ("supply_air_damper_percentage_command", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
      ("supply_air_flowrate_setpoint", (
          0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9
      )),
  )
  # pylint: enable=bad-continuation
  reader = controller_reader.ProtoReader(DATA_PATH)

  hr = histogram_reducer.HistogramReducer(
      histogram_parameters_tuples=histogram_parameters_tuples,
      reader=reader,
      normalize_reduce=True,
  )
  return hr
>>>>>>> copybara_push
