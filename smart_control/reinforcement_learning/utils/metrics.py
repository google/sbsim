"""Reinforcement learning metrics."""

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from smart_buildings.smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE

logger = logging.getLogger(__name__)


def get_trajectory(
    time_step: ts.TimeStep, current_action: policy_step.PolicyStep
) -> trajectory.Trajectory:
  """Get the trajectory for the current action and time step.

  Args:
      time_step: Current time step.
      current_action: Current action.

  Returns:
      Trajectory for the current action and time step.
  """
  observation = time_step.observation
  action = current_action.action
  policy_info = ()
  reward = time_step.reward
  discount = time_step.discount

  if time_step.is_first():
    return trajectory.first(observation, action, policy_info, reward, discount)

  if time_step.is_last():
    return trajectory.last(observation, action, policy_info, reward, discount)

  return trajectory.mid(observation, action, policy_info, reward, discount)


def compute_avg_return(
    environment: Any,
    policy: py_policy.PyPolicy,
    num_episodes: int = 1,
    time_zone: str = DEFAULT_TIME_ZONE,
    trajectory_observers: Optional[List[Callable]] = None,  # pylint: disable=g-bare-generic # TODO: use a more specific type hint if possible
    num_steps: int = 6,
) -> Tuple[float, List[List[Any]]]:
  """Computes the average return of the policy on the environment.

  Args:
      environment: Environment to evaluate on.
      policy: Policy to evaluate.
      num_episodes: Total number of episodes to run.
      time_zone: Time zone for timestamps.
      trajectory_observers: List of trajectory observers.
      num_steps: Number of steps to take per episode.

  Returns:
      Tuple of (average return, list of [simulation time, episode return]
      pairs).
  """
  total_return = 0.0
  return_by_simtime = []

  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    t0 = time.time()
    epoch = t0
    step_id = 0
    execution_times = []

    for _ in range(num_steps):
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)

      if trajectory_observers is not None:
        traj = get_trajectory(time_step, action_step)
        for observer in trajectory_observers:
          observer(traj)

      episode_return += time_step.reward
      t1 = time.time()
      dt = t1 - t0
      episode_seconds = t1 - epoch
      execution_times.append(dt)
      sim_time = environment.pyenv.envs[
          0
      ].current_simulation_timestamp.tz_convert(time_zone)

      return_by_simtime.append([sim_time, episode_return])

      logger.info(
          "[Step %d] [Sim Time: %s] [Reward: %.2f] [Return: %.2f] "
          "[Mean Step Time: %.2fs] [Episode Time: %.2fs]",
          step_id,
          sim_time.strftime("%Y-%m-%d %H:%M"),
          time_step.reward,
          episode_return,
          np.mean(execution_times),
          episode_seconds,
      )

      t0 = t1
      step_id += 1
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return, return_by_simtime
