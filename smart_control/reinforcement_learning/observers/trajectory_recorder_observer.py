# smart_control/reinforcement_learning/observers/trajectory_recorder_observer.py

import json
import logging
import os

from tf_agents.trajectories import trajectory as trajectory_lib

from smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_control.reinforcement_learning.visualization.trajectory_plotter import TrajectoryPlotter

logger = logging.getLogger(__name__)


class TrajectoryRecorderObserver(Observer):
  """Observer that records trajectory data for visualization.

  This observer saves information about the agent's actions, states, rewards,
  and timestamps during an episode for later visualization and generates plots.
  """

  def __init__(
      self,
      save_dir: str,
      environment=None,
      time_zone='US/Pacific',
      generate_plots=True,
  ):
    self._save_dir = save_dir
    self._environment = environment
    self._time_zone = time_zone
    self._episode_count = 0
    self._generate_plots = generate_plots

    # Create plots directory
    self._plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(self._plots_dir, exist_ok=True)

    # Initialize trajectory data containers
    self._reset_trajectory_data()

    # Get environment information
    self._num_timesteps_in_episode = self._environment.pyenv.envs[
        0
    ]._num_timesteps_in_episode

  def _reset_trajectory_data(self):
    """Reset the trajectory data containers."""
    self._actions = []
    self._rewards = []
    self._timestamps = []
    self._cumulative_reward = 0.0
    self._step_counts = []

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Record data at each step."""
    # Extract action from trajectory
    action = trajectory.action
    self._actions.append(action.tolist())

    # Extract reward and update cumulative reward
    reward = float(trajectory.reward)
    self._rewards.append(reward)
    self._cumulative_reward += reward

    # Get current simulation timestamp
    if hasattr(self._environment.pyenv.envs[0], 'current_simulation_timestamp'):
      sim_time = self._environment.pyenv.envs[0].current_simulation_timestamp
      if hasattr(sim_time, 'tz_convert'):
        sim_time = sim_time.tz_convert(self._time_zone)
      self._timestamps.append(str(sim_time))

    # Get current step count
    step_count = self._environment.pyenv.envs[0]._step_count
    self._step_counts.append(step_count)

    # Check if episode is done
    if trajectory.is_last():
      self._save_trajectory()

      # Generate plots if enabled
      if self._generate_plots:
        self._generate_plots_for_episode()

      self._reset_trajectory_data()
      self._episode_count += 1

  def _save_trajectory(self):
    """Save trajectory data to file."""
    trajectory_data = {
        'actions': self._actions,
        'rewards': self._rewards,
        'timestamps': self._timestamps,
        'step_counts': self._step_counts,
        'cumulative_reward': self._cumulative_reward,
        'episode_number': self._episode_count,
    }

    # Create filename and save
    episode_file = os.path.join(
        self._save_dir, f'episode_{self._episode_count}.json'
    )
    with open(episode_file, 'w') as f:
      json.dump(trajectory_data, f, indent=2)

    logger.info(
        f'Saved trajectory data for episode {self._episode_count} to'
        f' {episode_file}'
    )

  def _generate_plots_for_episode(self):
    """Generate plots for the current episode."""
    episode_num = self._episode_count

    # Generate action plot
    action_plot_path = os.path.join(
        self._plots_dir, f'episode_{episode_num}_action_plot.png'
    )
    TrajectoryPlotter.plot_actions(
        self._actions,
        action_plot_path,
        timestamps=self._timestamps if len(self._timestamps) <= 20 else None,
        title=f'Episode {episode_num}: Actions Over Time',
    )

    # Generate reward plot
    reward_plot_path = os.path.join(
        self._plots_dir, f'episode_{episode_num}_reward.png'
    )
    TrajectoryPlotter.plot_rewards(
        self._rewards,
        reward_plot_path,
        timestamps=self._timestamps if len(self._timestamps) <= 20 else None,
        title=f'Episode {episode_num}: Rewards Over Time',
    )

    # Generate cumulative reward plot
    cum_reward_plot_path = os.path.join(
        self._plots_dir, f'episode_{episode_num}_cum_reward.png'
    )
    TrajectoryPlotter.plot_cumulative_reward(
        self._rewards,
        cum_reward_plot_path,
        timestamps=self._timestamps if len(self._timestamps) <= 20 else None,
        title=f'Episode {episode_num}: Cumulative Reward Over Time',
    )

    logger.info(f'Generated plots for episode {episode_num}')

  def reset(self) -> None:
    """Reset the observer to its initial state."""
    self._reset_trajectory_data()
