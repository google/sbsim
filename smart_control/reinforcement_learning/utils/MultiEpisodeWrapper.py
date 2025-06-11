# -*- coding: utf-8 -*-
"""
Defines a simplified PyEnvironment wrapper that manages multiple environment
configurations, loading only one environment at a time and switching upon reset.
"""

import collections.abc  # Used for type hinting
import logging

# Import necessary TF-Agents components
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

# Configure logger for this module
logger = logging.getLogger(__name__)


class MultiEpisodeWrapper(py_environment.PyEnvironment):
  """
  A PyEnvironment wrapper that cycles through environment configurations ('scenarios')
  provided as file paths.

  Key characteristics:
  - Takes a list of configuration file paths.
  - Takes a function (`create_env_fn`) that can create an environment instance
    from a configuration path.
  - Only *one* underlying environment instance exists in memory at a time ('lazy' loading).
  - When an episode ends and `reset()` is called, it closes the current
    environment (if applicable), loads the *next* environment in a round-robin
    fashion using the next config path, and resets it.
  - Assumes all environments created from the different configs share the same
    action and observation specifications. The specs are determined from the
    first environment loaded.
  """

  def __init__(
      self,
      scenario_config_paths: collections.abc.Sequence[str],
      create_env_fn: collections.abc.Callable,
  ):
    """
    Initializes the lazy multi-scenario environment.

    Args:
        scenario_config_paths: A non-empty sequence (list, tuple) of string
                               paths pointing to environment configuration files.
        create_env_fn: A callable function that takes a single argument (a config path
                       from `scenario_config_paths`) and returns a fully constructed
                       `py_environment.PyEnvironment` instance.

    Raises:
        ValueError: If scenario_config_paths is empty.
        TypeError: If create_env_fn is not callable.
        Exception: If the first environment cannot be created or reset.
    """
    if not scenario_config_paths:
      raise ValueError("At least one scenario config path must be provided.")
    if not callable(create_env_fn):
      raise TypeError("`create_env_fn` must be a callable function.")

    logger.info(
        "Initializing LazyMultiScenarioPyEnvironment with"
        f" {len(scenario_config_paths)} config paths."
    )

    self._scenario_config_paths = list(scenario_config_paths)  # Store a copy
    self._num_paths = len(self._scenario_config_paths)
    self._create_env_fn = create_env_fn

    self._current_env_index = -1  # Start at -1 so the first load gets index 0
    self._current_env: py_environment.PyEnvironment | None = None
    self._state: ts.TimeStep | None = None

    # --- Load the first environment to determine specs ---
    try:
      self._load_and_reset_env()  # Loads env at index 0
      # Specs are determined from the first loaded environment
      self._action_spec = self._current_env.action_spec()
      self._observation_spec = self._current_env.observation_spec()
      self._time_step_spec = self._current_env.time_step_spec()
      logger.info(
          "Successfully loaded initial environment and determined specs."
      )
    except Exception as e:
      logger.exception(
          "Failed to load or reset the initial environment "
          f"(path: {self._scenario_config_paths[0]})."
      )
      raise RuntimeError("Could not initialize the first environment.") from e

    # Call the PyEnvironment base class initializer *after* specs are defined.
    super().__init__()

  def _load_and_reset_env(self):
    """Loads and resets the next environment in the sequence."""
    # Determine index of the next environment config path (round-robin)
    self._current_env_index = (self._current_env_index + 1) % self._num_paths
    config_path = self._scenario_config_paths[self._current_env_index]

    logger.info(
        f"Loading environment from config: {config_path} (index"
        f" {self._current_env_index})"
    )

    try:
      # Create the new environment instance
      self._current_env = self._create_env_fn(config_path)
      if not isinstance(self._current_env, py_environment.PyEnvironment):
        raise TypeError(
            "create_env_fn did not return a PyEnvironment instance for path:"
            f" {config_path}"
        )

      # Reset the newly loaded environment
      self._state = self._current_env.reset()
      logger.debug(
          "Successfully loaded and reset environment index"
          f" {self._current_env_index}"
      )

    except Exception as e:
      logger.exception(
          f"Failed to load or reset environment from config path: {config_path}"
      )
      # Propagate the error to indicate failure
      raise RuntimeError(
          f"Failed to load/reset environment from {config_path}"
      ) from e

  @property
  def current_environment(self) -> py_environment.PyEnvironment | None:
    """Returns the currently active underlying environment instance, if loaded."""
    # Added check for None in case called after close() or before init finishes
    return self._current_env

  @property
  def current_config_path(self) -> str | None:
    """Returns the config path of the currently active environment."""
    if (
        self._current_env_index >= 0
        and self._current_env_index < self._num_paths
    ):
      return self._scenario_config_paths[self._current_env_index]
    return None

  @property
  def _num_timesteps_in_episode(self):
    """Returns the number of timesteps in the current episode."""
    if self._current_env is not None:
      return self._current_env._num_timesteps_in_episode
    return None

  @property
  def _end_timestamp(self):
    """Returns the end timestamp of the current episode."""
    if self._current_env is not None:
      return self._current_env._end_timestamp
    return None

  @property
  def current_simulation_timestamp(self):
    """Returns the current simulation timestamp of the current episode."""
    if self._current_env is not None:
      return self._current_env.current_simulation_timestamp
    return None

  @property
  def _step_count(self):
    """Returns the step count of the current episode."""
    if self._current_env is not None:
      return self._current_env._step_count
    return None

  # --- PyEnvironment API Implementation ---

  def observation_spec(self):
    """Returns the observation spec (determined from the first environment)."""
    return self._observation_spec

  def action_spec(self):
    """Returns the action spec (determined from the first environment)."""
    return self._action_spec

  def time_step_spec(self):
    """Returns the time step spec (determined from the first environment)."""
    return self._time_step_spec

  def _reset(self) -> ts.TimeStep:
    """Closes the current environment, loads the next one, and resets it."""
    logger.debug(
        "Reset called. Closing current environment (index"
        f" {self._current_env_index})..."
    )
    # Close the previous environment, if it exists
    if self._current_env is not None:
      try:
        self._current_env.close()
        logger.debug("Previous environment closed.")
      except Exception as e:
        # Log error but continue, as we need to load the next one
        logger.error(
            "Error closing environment from path "
            f"'{self.current_config_path}': {e}"
        )
      finally:
        self._current_env = None  # Ensure it's marked as closed/gone

    # Load and reset the next environment in the sequence
    self._load_and_reset_env()

    # Return the initial state of the new environment
    return self._state

  def _step(self, action) -> ts.TimeStep:
    """Takes a step in the *currently loaded* underlying environment."""
    if self._current_env is None:
      # This should ideally not happen if used correctly within an RL loop
      raise RuntimeError(
          "Step called but no environment is currently loaded. Was reset()"
          " called?"
      )

    # Delegate the step call to the currently loaded environment.
    self._state = self._current_env.step(action)

    # If the episode ended, the next call should be reset(), which handles the switch.
    if self._state.is_last():
      logger.debug(
          f"Episode ended in environment index: {self._current_env_index}. "
          "Next reset call will switch environment."
      )

    return self._state

  def close(self):
    """Closes the currently loaded environment, if any."""
    logger.info("Close called on LazyMultiScenarioPyEnvironment.")
    if self._current_env is not None:
      try:
        logger.info(
            "Closing currently active environment (index"
            f" {self._current_env_index}, path: {self.current_config_path})"
        )
        self._current_env.close()
      except Exception as e:
        logger.error(
            "Error closing environment from path "
            f"'{self.current_config_path}': {e}"
        )
      finally:
        self._current_env = None  # Mark as closed
        self._current_env_index = -1  # Reset index state
    else:
      logger.info("No environment currently loaded, nothing to close.")

  def render(self, mode="rgb_array"):
    """Renders the *currently loaded* underlying environment."""
    if self._current_env is None:
      logger.warning("Render called but no environment is loaded.")
      return None  # Or raise an error

    try:
      return self.current_environment.render(mode)
    except Exception as e:
      logger.error(
          f"Failed to render environment index {self._current_env_index}: {e}"
      )
      return None  # Example: return None if rendering fails
