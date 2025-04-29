"""Utilities to reduce dimensionality of the observation space.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The observation space dimensionality of the real building is greater a thousand,
with multiple observations being of the same type. For example, in US-MTV-1055,
there are 127 VAVs each reporting zone_air_temperature_sensor values, creating
127 timelines of zone_air_temperature_sensor. One strategy to simplify
representation of the real building is to collapse these similar
timeseries into a central unit of measure, like a mean or median. This
reduced the observation space to less than 100, e.g., 79 in US-MTV-1055, which
makes training a regression model much more efficient.

The purpose of the reducer is to transform the full-width observation space
into a reduced space for model training and inference, and then to
generate (or expand) the full observation estimate.

The reduce() method reduces the input observation space for the model. The
expand() method estimates the full observation space for the agent.
"""

import abc
import collections
from typing import Any, Callable, Mapping, Sequence

import gin
import numpy as np
import pandas as pd

_STATS_FUNC_MAPPING = {'mean': np.mean, 'std': np.std, 'median': np.median}


@gin.configurable
def name_to_func(func_names: Sequence[str]) -> Sequence[Callable[..., float]]:
  """Maps function names to corresponding function callables."""
  funcs = []
  for func_name in func_names:
    if func_name not in _STATS_FUNC_MAPPING:
      raise ValueError(
          (
              'Requested func %s, is not in the list of supported ',
              'function names,',
          ),
          func_name,
      )
    funcs.append(_STATS_FUNC_MAPPING[func_name])

  return funcs


class BaseReducedSequence(metaclass=abc.ABCMeta):
  reduced_sequence: pd.DataFrame

  @abc.abstractmethod
  def expand(self) -> pd.DataFrame:
    """Expands the sequence into its original dimensionality."""


class BaseReducer(metaclass=abc.ABCMeta):
  """Base class for reducing state space into statistics."""

  @abc.abstractmethod
  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Converts the raw observation sequence into a reduced_sequence."""


@gin.configurable
class IdentityReducer(BaseReducer):
  """Pass-through 1:1 reducer without any reduction step, for compatibility."""

  class IdentityReducedSequence(BaseReducedSequence):

    def expand(self) -> pd.DataFrame:
      return self.reduced_sequence

  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Converts the raw observation sequence into a reduced_sequence."""
    rs = self.IdentityReducedSequence()
    rs.reduced_sequence = observation_sequence
    return rs


@gin.configurable
class StatsReducer(BaseReducer):
  """Reduces values to stats, and returns the median/average for each type."""

  class StatsReducedSequence(BaseReducedSequence):
    """Reduced sequence that expands with the stats value."""

    def __init__(
        self,
        passthrough_features: Sequence[Any],
        stats_funcs: Sequence[Callable[..., float]],
        feature_mapping: Mapping[str, Sequence[str]],
    ):
      self._passthrough_features = passthrough_features
      self._stats_funcs = stats_funcs
      self._feature_mapping = feature_mapping

    def expand(self) -> pd.DataFrame:
      current_observation_mapping = {}

      for feature in self._passthrough_features:
        if feature in self.reduced_sequence:
          current_observation_mapping[feature] = self.reduced_sequence[feature]

      for feature_list in self._feature_mapping.values():
        for feature in feature_list:
          feature_name = feature[-1]
          val = self.reduced_sequence[
              (feature_name, self._stats_funcs[0].__name__)
          ]
          current_observation_mapping[feature] = val

      return pd.DataFrame(current_observation_mapping)

  def __init__(
      self,
      passthrough_features: Sequence[Any],
      stats_funcs: Sequence[Callable[..., float]],
  ):
    """Initialization function.

    Args:
      passthrough_features: Feature names that should not be reduced.
      stats_funcs: Stats func for reduction, e.g., np.median, np.mean, etc.
    """
    self._passthrough_features = passthrough_features
    self._stats_funcs = stats_funcs

    if not self._stats_funcs:
      raise ValueError('Must provide at least one stats function.')

  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Converts the raw observation sequence into a reduced_sequence."""
    feature_mapping = self._get_feature_mapping(observation_sequence)

    def get_stats(
        observation_sequence, feature_mapping, feature_name, stats_funcs
    ):
      stats_dict = {}
      features = feature_mapping[feature_name]
      observation_subset = observation_sequence[features]

      for stats_func in stats_funcs:
        stats_dict[(feature_name, stats_func.__qualname__)] = stats_func(
            observation_subset, axis=1
        )
      return pd.DataFrame(stats_dict, index=observation_sequence.index)

    feature_stats = []
    for meta in self._passthrough_features:
      if meta in observation_sequence.columns:
        feature_stats.append(observation_sequence[meta])

    for feature in feature_mapping:
      feature_stats.append(
          get_stats(
              observation_sequence, feature_mapping, feature, self._stats_funcs
          )
      )

    reduced_sequence = pd.concat(feature_stats, axis=1)
    rs = self.StatsReducedSequence(
        passthrough_features=self._passthrough_features,
        stats_funcs=self._stats_funcs,
        feature_mapping=feature_mapping,
    )
    rs.reduced_sequence = reduced_sequence
    return rs

  def _get_feature_mapping(
      self, observation_sequence: pd.DataFrame
  ) -> Mapping[str, Sequence[str]]:
    feature_mapping = collections.defaultdict(list)
    for col in observation_sequence.columns:
      if col not in self._passthrough_features:
        feature_mapping[col[-1]].append(col)
    return feature_mapping
