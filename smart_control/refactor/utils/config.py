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
    DEFAULT_RENDERS_PATH,
    DEFAULT_EXPERIMENT_RESULTS_PATH
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
EXPERIMENT_RESULTS_PATH = DEFAULT_EXPERIMENT_RESULTS_PATH


def set_global_paths(
    data_path: Optional[str] = None,
    config_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    output_data_path: Optional[str] = None,
    root_dir: Optional[str] = None,
    experiment_results_path: Optional[str] = None
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
    if experiment_results_path is not None:
        EXPERIMENT_RESULTS_PATH = experiment_results_path
    
    # Create directories if they don't exist
    for path in [DATA_PATH, CONFIG_PATH, METRICS_PATH, OUTPUT_DATA_PATH, ROOT_DIR, EXPERIMENT_RESULTS_PATH]:
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
