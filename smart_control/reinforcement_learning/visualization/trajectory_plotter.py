"""Trajectory Plotter.

This module provides functions to plot trajectories of rl episodes.
"""

# smart_control/reinforcement_learning/visualization/trajectory_plotter.py

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
  """
  Utility class for generating plots from trajectory data.
  """

  @staticmethod
  def plot_actions(
      actions: List[List[float]],
      save_path: str,
      timestamps: List[str] = None,
      title: str = 'Actions Over Time',
  ) -> None:
    """
    Generate a plot showing action values over time.

    Args:
        actions: List of action values, where each action is a list of values
        save_path: Path to save the generated plot
        timestamps: Optional list of timestamp strings for x-axis
        title: Title for the plot
    """
    actions_array = np.array(actions)
    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = (
        range(len(actions)) if timestamps is None else range(len(timestamps))
    )
    action_dim = actions_array.shape[1] if len(actions_array.shape) > 1 else 1

    if action_dim == 1:
      ax.plot(x_values, actions_array, label='Action')
    else:
      for i in range(action_dim):
        ax.plot(x_values, actions_array[:, i], label=f'Action {i+1}')

    ax.set_xlabel('Time Step' if timestamps is None else 'Timestamp')
    ax.set_ylabel('Action Value')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Set x-ticks to timestamps if provided
    if timestamps is not None and len(timestamps) <= 20:
      # If too many timestamps, show a subset to avoid crowding
      plt.xticks(x_values, timestamps, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info('Saved action plot to %s', save_path)

  @staticmethod
  def plot_rewards(
      rewards: List[float],
      save_path: str,
      timestamps: List[str] = None,
      title: str = 'Rewards Over Time',
  ) -> None:
    """
    Generate a plot showing rewards at each time step.

    Args:
        rewards: List of reward values
        save_path: Path to save the generated plot
        timestamps: Optional list of timestamp strings for x-axis
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = (
        range(len(rewards)) if timestamps is None else range(len(timestamps))
    )
    ax.plot(
        x_values,
        rewards,
        label='Reward',
        marker='o',
        linestyle='-',
        markersize=4,
    )

    ax.set_xlabel('Time Step' if timestamps is None else 'Timestamp')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.grid(True)

    # Set x-ticks to timestamps if provided
    if timestamps is not None and len(timestamps) <= 20:
      plt.xticks(x_values, timestamps, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info('Saved reward plot to %s', save_path)

  @staticmethod
  def plot_cumulative_reward(
      rewards: List[float],
      save_path: str,
      timestamps: List[str] = None,
      title: str = 'Cumulative Reward Over Time',
  ) -> None:
    """
    Generate a plot showing the evolution of cumulative reward over time.

    Args:
        rewards: List of reward values
        save_path: Path to save the generated plot
        timestamps: Optional list of timestamp strings for x-axis
        title: Title for the plot
    """
    cumulative_rewards = np.cumsum(rewards)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = (
        range(len(rewards)) if timestamps is None else range(len(timestamps))
    )
    ax.plot(
        x_values, cumulative_rewards, label='Cumulative Reward', color='green'
    )

    ax.set_xlabel('Time Step' if timestamps is None else 'Timestamp')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(title)
    ax.grid(True)

    # Set x-ticks to timestamps if provided
    if timestamps is not None and len(timestamps) <= 20:
      plt.xticks(x_values, timestamps, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info('Saved cumulative reward plot to %s', save_path)
