"""Base class for all forecaster implementations.

This class defines the interface that BaseBuilding expects for getting
predictions about future action-invariant observations.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
