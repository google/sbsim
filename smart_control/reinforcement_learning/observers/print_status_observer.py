"""Reinforcement learning print status observer."""

import logging

import pandas as pd
from tf_agents.trajectories import trajectory as trajectory_lib

from smart_buildings.smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_buildings.smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE

logger = logging.getLogger(__name__)


class PrintStatusObserver(Observer):
  """Observer that prints status information.

  This observer prints information about the training progress, including
  rewards, execution time, and replay buffer size.
  """

  def __init__(
      self,
      status_interval_steps: int = 1,
      environment=None,
      replay_buffer=None,
      time_zone=DEFAULT_TIME_ZONE,
  ):
    self._counter = 0
    self._status_interval_steps = status_interval_steps
    self._environment = environment
    self._cumulative_reward = 0.0
    self._replay_buffer = replay_buffer
    self._time_zone = time_zone

    self._start_time = None
    self._num_timesteps_in_episode = self._environment.pyenv.envs[0]._num_timesteps_in_episode  # pylint: disable=line-too-long

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    reward = trajectory.reward
    self._cumulative_reward += reward
    self._counter += 1
    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._status_interval_steps == 0 and self._environment:

      execution_time = pd.Timestamp.now() - self._start_time
      mean_execution_time = execution_time.total_seconds() / self._counter

      sim_time = self._environment.pyenv.envs[0].current_simulation_timestamp.tz_convert(self._time_zone)  # pylint: disable=line-too-long
      percent_complete = int(100.0 * (self._counter / self._num_timesteps_in_episode))  # pylint: disable=line-too-long

      rb_string = ""
      if self._replay_buffer is not None:
        rb_size = self._replay_buffer.num_frames()
        rb_string = f"Replay Buffer Size: {rb_size}"

      logger.info(
          "[Step %d of %d %d%%] [Sim Time: %s] [Reward: %.2f] "
          "[Cum Reward: %.2f]",
          self._environment.pyenv.envs[0]._step_count,
          self._num_timesteps_in_episode,
          percent_complete,
          sim_time.strftime("%Y-%m-%d %H:%M"),
          reward,
          self._cumulative_reward,
      )

      logger.info(
          "[Exec Time: %s] [Mean Exec Time: %.2fs] [%s]",
          execution_time,
          mean_execution_time,
          rb_string,
      )

  def reset(self) -> None:
    """Reset the observer to its initial state."""
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
