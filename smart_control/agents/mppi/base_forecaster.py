"""Base class for all forecaster implementations.

This class defines the interface that BaseBuilding expects for getting
predictions about future action-invariant observations.
"""

import abc
from typing import Dict, Optional, Sequence
import pandas as pd


class BaseForecaster(abc.ABC):
  """Abstract base class for all forecaster implementations.

  This class defines the interface that the MPPIPolicy expects for getting
  predictions about future action-invariant observations.
  """

  @abc.abstractmethod
  def predict(
      self, timestamp: pd.Timestamp, features: Sequence[str]
  ) -> Optional[Dict[str, float]]:
    """For a given future timestamp, return a dictionary of predicted values.

    Args:
        timestamp: The future timestamp to generate a forecast for.
        features: A list of feature names to be forecasted.

    Returns:
        A dictionary mapping feature names to their forecasted values,
        or None if a forecast cannot be made (e.g., error or missing data).
    """
