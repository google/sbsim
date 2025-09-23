"""Base observer interface for all RL observers.

This module defines the Observer abstract class that all RL observers should
implement.
"""

import abc

from tf_agents.trajectories import trajectory as trajectory_lib


class Observer(abc.ABC):
  """Abstract base class for all observers.

  Observers are objects that monitor the training process, collect metrics,
  and visualize the agent's behavior. They are called with trajectories
  during data collection.
  """

  @abc.abstractmethod
  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Process a trajectory.

    Args:
        trajectory: A trajectory to process.
    """
    pass

  @abc.abstractmethod
  def reset(self) -> None:
    """Reset the observer to its initial state.

    This method is called when a new episode starts.
    """
    pass
