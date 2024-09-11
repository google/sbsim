"""Histogram Reducer for RegressionBuilding.

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


The objective of the histogram reducer is to compress a very wide
multivariate timeseries with minimal data loss. The current control agents
don't really benefit from knowing the temperature (etc.) of each zone, but
simply need to know that some zones are below of above setpoints. As such,
representing each zone as a separate timeseries is rather inefficient.

Reduce function converts a feature from individual timeseries into a histogram.
For exammple, devices d1, d2 have a zone_air_temperature timeseries,
the histogram reducer converts the timeseries into a counts on temperature
bins, like 70, 71, 72, etc. and assigns a count to the bin. This reduces
the dimensionality into a more compressed format if the number of the devices
exceeds the number of bins.

The histogram operation also caps the counts to the max and min values, so
the lower and upper ends represent less than or equal to the lowest bin value,
and greater than or equal to the highest bin value, respectively. IOW,
For internal bins i = 1...N-2, assign v to bin i if, bin[i] <= v < bin[i+1].
For first bin, assign v to bin 0 if v < bin[1]. For the last bin, assign v
to bin N-1 if bin[N-1] <= v.

Expand function takes the counts in the histogram and reconstructs lossy
timeseries for each device. For example, suppose a measurement of 72.7 is
assigned to bin 72, then the approximate measurement would be the lower
bound on the bin (i.e., 72.0).
"""

import collections
from typing import Dict, List, Mapping, Sequence, Union

from absl import logging
import gin
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import reader_lib
from smart_control.utils.reducer import BaseReducedSequence
from smart_control.utils.reducer import BaseReducer

Feature = str  # Measurement name
Device = str  # Device Identity
Value = float  # Measurement value
# For each bin a list of assigned devices
HistogramAssignment = Sequence[List[Device]]
# Count of devices assigned to each histogram bin.
HistogramCounts = Sequence[Union[float, int]]
# Approximate measurements taken from the histogram
HistogramExpansion = Mapping[tuple[Device, Feature], Value]
# Numeric bins for a histogram
HistogramBins = np.ndarray
# For each feature, a list of bins
HistogramParameters = Dict[Feature, HistogramBins]


def assign_devices_to_bins(
    feature_name: Feature,
    bins: HistogramBins,
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> HistogramAssignment:
  """For feature, create a histogram assignment from an ObservationResponse.

  For internal bins i = 1...N-2, assign v to bin i if, bin[i] <= v < bin[i+1].
  For first bin, assign v to bin 0 if v < bin[1].
  For the last bin, assign v to bin N-1 if bin[N-1] <= v.

  Args:
    feature_name: The feature or measurement desired
    bins: The assigned numeric bins, monotonically increasing.
    observation_response: The Observation response for the histogram.

  Returns:
    A jagged array with outer dim for each bin, and inner array with device ids.
  """
  # Create a an eply assignment as a list of lists, one list per bin.
  assignment = [[] for _ in range(len(bins))]

  for (
      single_observation_response
  ) in observation_response.single_observation_responses:
    if (
        single_observation_response.single_observation_request.measurement_name
        != feature_name
    ):
      continue

    v = single_observation_response.continuous_value

    v_bin_index = np.argwhere(np.concatenate([[True], v >= bins[1:]])).max()
    assignment[v_bin_index].append(
        single_observation_response.single_observation_request.device_id
    )

  return assignment


def approximate_values_from_histogram_assignment(
    measurement_name: Feature,
    assignment: HistogramAssignment,
    bins: HistogramBins,
) -> HistogramExpansion:
  """Creates approximate measurements from a histogram assignment.

  Args:
    measurement_name: the measurement name.
    assignment: for each bin, the list of assigned devices.
    bins: the values associated with each bin.

  Returns:
    A mapping of (device_id, measurement_name): bin-assigned value
  """
  assigned_values = {}

  for i, _ in enumerate(bins):
    for device_name in assignment[i]:
      assigned_values[(device_name, measurement_name)] = bins[i]

  return assigned_values


def get_clipped_histogram(
    measurements: np.ndarray, bins: np.ndarray, clip: bool = True
) -> np.ndarray:
  """Creates an array to a histogram, with min/max clipping."""

  # TODO(sipple): Consolidate the logic with assign_devices_to_bins().
  cbins = np.append(bins, max(bins))
  if clip:
    measurements = np.clip(measurements, min(cbins), max(cbins))

  return np.histogram(measurements, bins=cbins)[0].astype(np.float32)


def reassign_nodes(
    assignment: HistogramAssignment, histogram_counts_next: HistogramCounts
) -> HistogramAssignment:
  """Takes a current assignment and shifts it to match the HistogramCounts.

  Moves devices from one bin to another to match the next histogram counts as
  efficiently as possible (i.e., moves a devices from the closest possible
  current bin assignment.)

  Args:
    assignment: The current assignment of devices to bins.
    histogram_counts_next: counts assigned by the next histogram step.

  Returns:
    The next HistogramAssignment that matches the histogram_counts_next.
  """
  node_counts_current = [len(n) for n in assignment]

  if len(node_counts_current) != len(histogram_counts_next):
    raise ValueError(
        "Number of bins don't match. node_counts_current"
        f" {len(node_counts_current)} and histogram_counts_next"
        f" {len(histogram_counts_next)}."
    )

  if np.sum(node_counts_current) != np.sum(histogram_counts_next):
    raise ValueError(
        f"Assignment has {np.sum(node_counts_current)} nodes, but"
        f" histogram_counts_next has {np.sum(histogram_counts_next)} nodes. The"
        " counts must match."
    )

  for i, _ in enumerate(node_counts_current):
    # If the current assignment at i has more devices than the next assignment,
    # shift the extra device by one bin to the right, until the number
    # of devices assigned to the bin match the next assignment.
    while node_counts_current[i] > histogram_counts_next[i]:
      node_move = assignment[i].pop(0)
      assignment[i + 1].append(node_move)
      node_counts_current = [len(n) for n in assignment]

    # If the current assignment at i has fewer than the next assignment,
    # find the nearest device j: j > i and move that device from j to i.
    while node_counts_current[i] < histogram_counts_next[i]:
      for j in range(i + 1, len(node_counts_current)):
        if node_counts_current[j] > 0:
          node_move = assignment[j][0]
          assignment[j].remove(node_move)
          assignment[i].append(node_move)
          node_counts_current = [len(n) for n in assignment]
          break
  return assignment


@gin.configurable
class HistogramReducer(BaseReducer):
  """Implementation of the HistogramReducer.

  The objective of the histogram reducer is to compress a very wide
  multivariate timeseries with minimal data loss. The current control agents
  don't really benefit from knowing the temperature (etc.) of each zone, but
  simply need to know that some zones are below of above setpoints. As such,
  representing each zone as a separate timeseries is rather inefficient.

  Reduce function converts a feature from timeseries into a histogram.
  For exammple, devices d1, d2 have a zone_air_temperature timeseries,
  the histogram reducer converts the timeseries into a counts on temperature
  bins, like 70, 71, 72, etc. and assigns a count to the bin. This reduces
  the dimensionality into a more compressed format if the number of the devices
  exceeds the number of bins.

  The histogram operation also caps the counts to the max and min values, so
  the lower and upper ends represent less than or equal to the lowest bin value,
  and greater than or equal to the highest bin value, respectively. IOW,
  For internal bins i = 1...N-2, assign v to bin i if, bin[i] <= v < bin[i+1].
  For first bin, assign v to bin 0 if v < bin[1]. For the last bin, assign v
  to bin N-1 if bin[N-1] <= v.

  Expand function takes the counts in the histogram and reconstructs lossy
  timeseries for each device. For example, suppose a measurement of 72.7 is
  assigned to bin 72, then the approximate measurement would be the lower
  bound on the bin (i.e., 72.0).
  """

  def __init__(
      self,
      histogram_parameters_tuples: Sequence[tuple[str, np.ndarray]],
      reader: reader_lib.BaseReader,
      normalize_reduce: bool = False,
  ):
    self._normalize_reduce = normalize_reduce
    self._histogram_parameters: HistogramParameters = {
        p[0]: np.array(p[1]) for p in histogram_parameters_tuples
    }
    logging.info("histogram parameters: %s", self._histogram_parameters)
    observation_responses = reader.read_observation_responses(
        start_time=pd.Timestamp.min, end_time=pd.Timestamp.max
    )
    initial_observation_response = observation_responses[0]

    self._histogram_assignments = {}
    for feature_name in self._histogram_parameters:
      self._histogram_assignments[feature_name] = assign_devices_to_bins(
          feature_name,
          self._histogram_parameters[feature_name],
          initial_observation_response,
      )
    logging.info("histogram assignments: %s", self._histogram_assignments)

  class HistogramReducedSequence(BaseReducedSequence):
    """ReducedSequence that returns the histogram."""

    def __init__(
        self,
        histogram_parameters: HistogramParameters,
        histogram_assignments: Dict[Feature, HistogramAssignment],
        passthrough_sequence: pd.DataFrame,
        reduced_sequence: pd.DataFrame,
    ):
      self._histogram_parameters = histogram_parameters
      self._passthrough_sequence = passthrough_sequence
      self._histogram_assignments = histogram_assignments
      self.reduced_sequence = reduced_sequence

    def expand(self) -> pd.DataFrame:
      updates = collections.defaultdict(list)
      indexes = []

      def _fix_approximate_assignments(
          histogram_counts_current: Sequence[float],
          histogram_counts_next: list[float],
      ) -> Sequence[float]:
        """Adjusts the approximate counts to match the current counts."""

        diff = np.sum(histogram_counts_current) - np.sum(histogram_counts_next)

        if np.abs(diff) > 0:
          while np.abs(diff) > 0:
            max_ix = np.argmax(histogram_counts_next)
            histogram_counts_next[max_ix] += np.sign(diff)
            diff = np.sum(histogram_counts_current) - np.sum(
                histogram_counts_next
            )
        return histogram_counts_next

      def _count_bin_assignments(
          current_histogram_assignment: HistogramAssignment,
      ) -> Sequence[int]:
        """Returns the counts per bin from the assignment."""
        return [len(n) for n in current_histogram_assignment]

      for idx, row in self.reduced_sequence.iterrows():
        indexes.append(idx)

        for measurement_name in self._histogram_parameters:
          # Get the bin values.
          bins = self._histogram_parameters[measurement_name]
          # Create a tuple for the count in each bin
          node_counts_tuple = [
              (float(col[1].replace("h_", "")), row[col])
              for col in self.reduced_sequence.columns
              if col[0] == measurement_name
          ]
          if not node_counts_tuple:
            continue

          # Sort the list of tuples by the bin value.
          node_counts_tuple.sort(key=lambda a: a[0])

          # Now just get the counts of the bins to make the next bin assignment.
          histogram_counts_next = [
              max(0, int(tup[1])) for tup in node_counts_tuple
          ]

          current_histogram_assignment = self._histogram_assignments[
              measurement_name
          ]

          histogram_counts_current = _count_bin_assignments(
              current_histogram_assignment
          )

          histogram_counts_next = _fix_approximate_assignments(
              histogram_counts_current, histogram_counts_next
          )
          # Now reassign the devices to the next bins.
          next_histogram_assignment = reassign_nodes(
              current_histogram_assignment, histogram_counts_next
          )

          # From the new assignment, get measurements based on the assigned bin.
          next_assigned_measurements = (
              approximate_values_from_histogram_assignment(
                  measurement_name, next_histogram_assignment, bins
              )
          )

          self._histogram_assignments[measurement_name] = (
              next_histogram_assignment
          )

          for measurement in next_assigned_measurements:
            updates[measurement].append(next_assigned_measurements[measurement])

      df = pd.DataFrame(updates, index=indexes)

      # Add in "passthough" features that are not histogrammed.
      if self._passthrough_sequence is not None:
        # Prefer the columns in the reduced sequence over the
        # passthrough values.
        cols_in_reduced_and_passthrough = list(
            set(self._passthrough_sequence.columns).intersection(
                self.reduced_sequence.columns
            )
        )
        cols_in_passthrough_only = list(
            set(self._passthrough_sequence.columns).difference(
                self.reduced_sequence.columns
            )
        )

        df = pd.concat(
            [
                df,
                self._passthrough_sequence[cols_in_passthrough_only],
                self.reduced_sequence[cols_in_reduced_and_passthrough],
            ],
            axis=1,
            join="inner",
        )

      return df

    @property
    def feature_device_assignments(
        self,
    ) -> Dict[Feature, HistogramAssignment]:
      return self._histogram_assignments

  @property
  def histogram_parameters(self) -> HistogramParameters:
    return self._histogram_parameters

  def _get_passthrough_sequence(
      self, observation_sequence: pd.DataFrame
  ) -> pd.DataFrame:
    """Returns a dataframe with the features that are not histogrammed."""

    passthrough_columns = []
    for tup in observation_sequence.columns:
      if not isinstance(tup, tuple):
        passthrough_columns.append(tup)
        continue
      elif len(tup) == 2:
        device, measurement = tup
      else:
        _, device, measurement = tup
      if measurement not in self._histogram_parameters.keys():
        passthrough_columns.append((device, measurement))

    return observation_sequence[np.array(passthrough_columns, dtype=object)]

  def _get_reduced_sequence(
      self,
      observation_sequence: pd.DataFrame,
      feature_mapping: Mapping[str, Sequence[str]],
  ) -> Sequence[pd.DataFrame]:
    """Converts the raw features into histogram features."""
    reduced_feature_dfs = []
    for reduced_feature, bins in self._histogram_parameters.items():
      reduced_feature_columns = feature_mapping[reduced_feature]
      # Now compute the histogram
      if reduced_feature_columns:
        columns_indexes = [(reduced_feature, "h_%.2f" % v) for v in bins]
        df = pd.DataFrame(columns=columns_indexes)
        for idx, row in observation_sequence.iterrows():
          # Convert all the measurements of the same feature into an array.
          measurements = np.array(row[reduced_feature_columns])

          chist = get_clipped_histogram(
              measurements=measurements, bins=bins, clip=True
          )
          df.loc[idx] = chist
          if self._normalize_reduce:
            df.loc[idx] /= np.sum(chist)
        reduced_feature_dfs.append(df)
    return reduced_feature_dfs

  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Converts the raw observation sequence into a reduced_sequence."""

    passthrough_sequence = self._get_passthrough_sequence(observation_sequence)

    feature_mapping = self._get_feature_mapping(observation_sequence)

    reduced_feature_dfs = self._get_reduced_sequence(
        observation_sequence, feature_mapping
    )

    # Join the passthrough and the rediced sequences into a single dataframe.
    reduced_sequence = passthrough_sequence
    if reduced_feature_dfs:
      df_hist = pd.concat(reduced_feature_dfs, axis=1)

      reduced_sequence = pd.concat([reduced_sequence, df_hist], axis=1)

    rs = self.HistogramReducedSequence(
        self._histogram_parameters,
        self._histogram_assignments,
        passthrough_sequence,
        reduced_sequence,
    )

    return rs

  def _get_feature_mapping(
      self, observation_sequence: pd.DataFrame
  ) -> Mapping[Feature, Sequence[Device]]:
    feature_mapping = collections.defaultdict(list)
    for col in observation_sequence.columns:
      if col[-1] in self._histogram_parameters:
        feature_mapping[col[-1]].append(col)
    return feature_mapping
