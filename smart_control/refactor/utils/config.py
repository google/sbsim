import os
import gin
import numpy as np
from typing import Any, Optional

from smart_control.refactor.utils.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_OUTPUT_DATA_PATH,
    DEFAULT_ROOT_DIR,
    DEFAULT_RENDERS_PATH
)
from smart_control.utils import histogram_reducer
from smart_control.utils import controller_reader

# Global path variables - can be overridden by config
DATA_PATH = DEFAULT_DATA_PATH
CONFIG_PATH = DEFAULT_CONFIG_PATH
METRICS_PATH = DEFAULT_METRICS_PATH
OUTPUT_DATA_PATH = DEFAULT_OUTPUT_DATA_PATH
ROOT_DIR = DEFAULT_ROOT_DIR
RENDERS_PATH = DEFAULT_RENDERS_PATH


def set_global_paths(
    data_path: Optional[str] = None,
    config_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    output_data_path: Optional[str] = None,
    root_dir: Optional[str] = None,
) -> None:
    """Set global path variables.
    
    Args:
        data_path: Path to data directory.
        config_path: Path to config directory.
        metrics_path: Path to metrics directory.
        output_data_path: Path to output data directory.
        root_dir: Root directory.
    """
    global DATA_PATH, CONFIG_PATH, METRICS_PATH, OUTPUT_DATA_PATH, ROOT_DIR
    
    if data_path is not None:
        DATA_PATH = data_path
    if config_path is not None:
        CONFIG_PATH = config_path
    if metrics_path is not None:
        METRICS_PATH = metrics_path
    if output_data_path is not None:
        OUTPUT_DATA_PATH = output_data_path
    if root_dir is not None:
        ROOT_DIR = root_dir
    
    # Create directories if they don't exist
    for path in [DATA_PATH, CONFIG_PATH, METRICS_PATH, OUTPUT_DATA_PATH, ROOT_DIR]:
        os.makedirs(path, exist_ok=True)


def remap_filepath(filepath: str) -> str:
    """Remap filepath based on environment variables or other rules.
    
    Args:
        filepath: Original filepath.
        
    Returns:
        Remapped filepath.
    """
    # This function can be extended to handle more complex filepath remapping
    return filepath


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


def load_environment(gin_config_file: str) -> Any:
    """Load environment from gin config file.
    
    Args:
        gin_config_file: Path to gin config file.
        
    Returns:
        Environment.
    """
    # Import here to avoid circular imports
    from smart_control.environment.environment import Environment
    
    with gin.unlock_config():
        gin.clear_config()
        gin.parse_config_file(gin_config_file)
        return Environment()  # pylint: disable=no-value-for-parameter


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
