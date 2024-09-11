"""Tests for reducer.

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

"""

from absl.testing import absltest
import numpy as np
import pandas as pd
from smart_control.utils import reducer


class ReducerTest(absltest.TestCase):

  def _get_test_observation_sequence(self):
    return pd.DataFrame({
        ('a', 'b'): [1, 2, 3],
        ('d0', 'm0'): [76.0, 76.2, 75.0],
        ('d1', 'm0'): [74.1, 75.2, 77.0],
        ('d2', 'm0'): [73.1, 73.2, 74.0],
        ('d1', 'm1'): [0.2, 0.22, 0.23],
    })

  def _get_test_median_reduced_sequence(self):
    return pd.DataFrame({
        ('a', 'b'): [1, 2, 3],
        ('m0', 'median'): [74.1, 75.2, 75.0],
        ('m1', 'median'): [0.2, 0.22, 0.23],
    })

  def _get_test_median_expanded_sequence(self):
    return pd.DataFrame({
        ('a', 'b'): [1, 2, 3],
        ('d0', 'm0'): [74.1, 75.2, 75.0],
        ('d1', 'm0'): [74.1, 75.2, 75.0],
        ('d2', 'm0'): [74.1, 75.2, 75.0],
        ('d1', 'm1'): [0.2, 0.22, 0.23],
    })

  def test_identity_reducer(self):
    observation_sequence = self._get_test_observation_sequence()
    identity_reducer = reducer.IdentityReducer()
    rs = identity_reducer.reduce(observation_sequence)
    pd.testing.assert_frame_equal(rs.reduced_sequence, observation_sequence)
    pd.testing.assert_frame_equal(rs.expand(), observation_sequence)

  def test_median_reducer_reduce(self):
    observation_sequence = self._get_test_observation_sequence()
    median_reduced_sequence = self._get_test_median_reduced_sequence()
    passthrough_features = [('a', 'b')]
    stats_funcs = [np.median]
    stats_reducer = reducer.StatsReducer(
        passthrough_features=passthrough_features, stats_funcs=stats_funcs
    )
    rs = stats_reducer.reduce(observation_sequence)
    pd.testing.assert_frame_equal(rs.reduced_sequence, median_reduced_sequence)

  def test_median_reducer_expand(self):
    observation_sequence = self._get_test_observation_sequence()
    median_expanded_sequence = self._get_test_median_expanded_sequence()
    passthrough_features = [('a', 'b')]
    stats_funcs = [np.median]
    stats_reducer = reducer.StatsReducer(
        passthrough_features=passthrough_features, stats_funcs=stats_funcs
    )
    reduced_sequence = stats_reducer.reduce(observation_sequence)
    expanded_sequence = reduced_sequence.expand()
    pd.testing.assert_frame_equal(expanded_sequence, median_expanded_sequence)

  def test_empty_stats_funcs(self):
    passthrough_features = [('a', 'b')]
    with self.assertRaises(ValueError):
      _ = reducer.StatsReducer(
          passthrough_features=passthrough_features, stats_funcs=[]
      )

  def test_bad_func_reducer_reduce(self):
    def bad_stats_func(a, axis=None, dtype=None, out=None):
      raise ValueError('Bad stats function')

    observation_sequence = self._get_test_observation_sequence()
    passthrough_features = [('a', 'b')]

    stats_funcs = [bad_stats_func]
    stats_reducer = reducer.StatsReducer(
        passthrough_features=passthrough_features, stats_funcs=stats_funcs
    )
    with self.assertRaises(ValueError):
      _ = stats_reducer.reduce(observation_sequence)


if __name__ == '__main__':
  absltest.main()
