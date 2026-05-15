"""Reinforcement learning composite observer."""

from typing import Sequence

from tf_agents.trajectories import trajectory as trajectory_lib

from smart_buildings.smart_control.reinforcement_learning.observers.base_observer import Observer


class CompositeObserver(Observer):
  """Observer that combines multiple observers.

  This observer calls all of its constituent observers whenever it is called.
  It provides a convenient way to use multiple observers together.
  """

  def __init__(self, observers: Sequence[Observer]):
    """Initialize the observer.

    Args:
        observers: A sequence of observers to combine.
    """
    self._observers = list(observers)

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Process a trajectory with all observers.

    Args:
        trajectory: A trajectory to process.
    """
    for observer in self._observers:
      observer(trajectory)

  def reset(self) -> None:
    """Reset all observers."""
    for observer in self._observers:
      observer.reset()

  def close(self) -> None:
    """Close all observers."""
    for observer in self._observers:
      observer.close()

  def add_observer(self, observer: Observer) -> None:
    """Add an observer to the composite.

    Args:
        observer: The observer to add.
    """
    self._observers.append(observer)

  def remove_observer(self, observer: Observer) -> None:
    """Remove an observer from the composite.

    Args:
        observer: The observer to remove.
    """
    if observer in self._observers:
      self._observers.remove(observer)
