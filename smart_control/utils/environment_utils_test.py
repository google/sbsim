from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_normalization_pb2
from smart_buildings.smart_control.utils import bounded_action_normalizer
from smart_buildings.smart_control.utils import environment_utils

BoundedActionNormalizer = bounded_action_normalizer.BoundedActionNormalizer
ContinuousVariableInfo = smart_control_normalization_pb2.ContinuousVariableInfo


class GinUtilsTest(absltest.TestCase):

  def test_local_time(self):
    self.assertEqual(
        pd.Timedelta('07:12:01'), environment_utils.local_time('07:12:01')
    )

  def test_to_timestamp(self):
    self.assertEqual(
        pd.Timestamp('2021-08-13 14:01:33'),
        environment_utils.to_timestamp('2021-08-13 14:01:33'),
    )

  def test_set_observation_normalization_constants(self):
    observation = environment_utils.set_observation_normalization_constants(
        field_id='field_id', sample_mean=1.0, sample_variance=2.0
    )
    self.assertIsInstance(observation, ContinuousVariableInfo)
    self.assertEqual(observation.id, 'field_id')
    self.assertEqual(observation.sample_mean, 1.0)
    self.assertEqual(observation.sample_variance, 2.0)

  def test_set_action_normalization_constants(self):
    normalizer = environment_utils.set_action_normalization_constants(
        min_native_value=0.0,
        max_native_value=1.0,
        min_normalized_value=-1.0,
        max_normalized_value=1.0,
    )
    self.assertIsInstance(normalizer, BoundedActionNormalizer)
    self.assertEqual(normalizer.min_native_value, 0.0)
    self.assertEqual(normalizer.max_native_value, 1.0)
    self.assertEqual(normalizer.min_normalized_value, -1.0)
    self.assertEqual(normalizer.max_normalized_value, 1.0)


if __name__ == '__main__':
  absltest.main()
