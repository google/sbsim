"""Reinforcement learning time utils."""

import numpy as np


def time_from_sin_cos(sin_theta: float, cos_theta: float) -> float:
  """Converts sin/cos representation to radians (time angle)."""
  if sin_theta >= 0:
    return (
        cos_theta >= 0 and np.arccos(cos_theta) or np.pi - np.arcsin(sin_theta)
    )
  else:
    return (
        cos_theta < 0
        and np.pi - np.arcsin(sin_theta)
        or 2 * np.pi - np.arccos(cos_theta)
    )


def to_dow(sin_theta: float, cos_theta: float) -> int:
  """Converts sin/cos to day of week (0-6)."""
  theta = time_from_sin_cos(sin_theta, cos_theta)
  return int(np.floor(7 * theta / (2 * np.pi)))


def to_hod(sin_theta: float, cos_theta: float) -> int:
  """Converts sin/cos to hour of day (0-23)."""
  theta = time_from_sin_cos(sin_theta, cos_theta)
  return int(np.floor(24 * theta / (2 * np.pi)))
