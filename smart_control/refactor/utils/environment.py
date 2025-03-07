import gin
import numpy as np
import os
from typing import Any

from smart_control.simulator.weather_controller import ReplayWeatherController
from smart_control.simulator.stochastic_convection_simulator import StochasticConvectionSimulator
from smart_control.simulator.building import MaterialProperties
from smart_control.simulator.air_handler import AirHandler
from smart_control.simulator.boiler import Boiler
from smart_control.simulator.hvac_floorplan_based import FloorPlanBasedHvac
from smart_control.utils.environment_utils import to_timestamp
from smart_control.simulator.tf_simulator import TFSimulator
from smart_control.simulator.simulator_building import SimulatorBuilding
from smart_control.reward.setpoint_energy_carbon_regret import SetpointEnergyCarbonRegretFunction
from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
from smart_control.utils.observation_normalizer import StandardScoreObservationNormalizer
from smart_control.utils.controller_writer import ProtoWriterFactory

from smart_control.refactor.utils.config import DATA_PATH, METRICS_PATH, remap_filepath
from smart_control.refactor.utils.constants import DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT
from smart_control.environment.environment import Environment
from smart_control.utils import histogram_reducer
from smart_control.utils import controller_reader


@gin.configurable
def get_histogram_path() -> str:
    """Get path to histogram data.
    
    Returns:
        Path to histogram data.
    """
    return DATA_PATH


@gin.configurable
def get_reset_temp_values() -> np.ndarray:
    """Get reset temperature values.
    
    Returns:
        Reset temperature values.
    """
    reset_temps_filepath = remap_filepath(
        os.path.join(DATA_PATH, "reset_temps.npy")
    )
    
    return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path() -> str:
    """Get path to zone data.
    
    Returns:
        Path to zone data.
    """
    return remap_filepath(
        os.path.join(DATA_PATH, "double_resolution_zone_1_2.npy")
    )


@gin.configurable
def get_metrics_path() -> str:
    """Get path to metrics.
    
    Returns:
        Path to metrics.
    """
    return os.path.join(METRICS_PATH, "metrics")


@gin.configurable
def get_weather_path() -> str:
    """Get path to weather data.
    
    Returns:
        Path to weather data.
    """
    return remap_filepath(os.path.join(DATA_PATH, "local_weather_moffett_field_20230701_20231122.csv"))


@gin.configurable
def get_histogram_reducer() -> Any:
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


def load_environment(gin_config_file: str):
    """Returns an Environment from a config file."""
    # Global definition is required by Gin library to instantiate Environment.
    # global environment  # pylint: disable=global-variable-not-assigned

    with gin.unlock_config():
        gin.clear_config()
        gin.parse_config_file(gin_config_file)
        return Environment()  # pylint: disable=no-value-for-parameter


def create_and_setup_environment(
    gin_config_file: str,
    metrics_path: str = None,
    occupancy_normalization_constant: float = DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT
):
    """Creates and sets up the environment."""
    env = load_environment(gin_config_file)
    env._metrics_path = metrics_path
    env._occupancy_normalization_constant = occupancy_normalization_constant
    
    return env
    