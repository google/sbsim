"""Tests for action_normalizer.

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
from absl.testing import parameterized
import numpy as np
from smart_control.utils import bounded_action_normalizer
from tf_agents import specs


class ActionNormalizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('min_native_value', -1, 200),
      ('mid_value', 0, 250),
      ('mid_positive_value', 0.5, 275),
      ('max_native_value', 1, 300),
  )
  def test_default_bounded_action_setpoint_value(
      self, agent_action, expected_value
  ):
    min_native_value = 200
    max_native_value = 300

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value, max_native_value=max_native_value
    )

    output = handler.setpoint_value(np.array(agent_action))

    self.assertEqual(output, expected_value)

  @parameterized.named_parameters(
      ('min_native_value', 0, 200),
      ('mid_value', 0.5, 250),
      ('mid_positive_value', 0.75, 275),
      ('max_native_value', 1, 300),
  )
  def test_normalized_range_bounded_action_setpoint_value(
      self, agent_action, expected_value
  ):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = 0
    max_normalized_value = 1

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    output = handler.setpoint_value(np.array(agent_action))

    self.assertEqual(output, expected_value)

  @parameterized.named_parameters(
      ('min_native_value', 200, 0),
      ('mid_value', 250, 0.5),
      ('mid_positive_value', 275, 0.75),
      ('max_native_value', 300, 1),
  )
  def test_normalized_range_bounded_agent_setpoint_value(
      self, agent_action, expected_value
  ):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = 0
    max_normalized_value = 1

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    output = handler.agent_value(np.array(agent_action))

    self.assertEqual(output, expected_value)

  def test_normalized_range_bounded_agent_setpoint_above(self):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = 0
    max_normalized_value = 1
    agent_action = 301

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    with self.assertRaises(ValueError):
      _ = handler.agent_value(np.array(agent_action))

  def test_normalized_range_bounded_agent_setpoint_below(self):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = 0
    max_normalized_value = 1
    agent_action = 199

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    with self.assertRaises(ValueError):
      _ = handler.agent_value(np.array(agent_action))

  @parameterized.named_parameters(
      ('default_range_below', -1, 1, -2),
      ('default_range_above', -1, 1, 3),
      ('half_range_below', 0, 1, -0.5),
      ('half_range_above', 0, 1, 1.3),
  )
  def test_normalized_range_setpoint_value_raises_error(
      self, min_normalized, max_normalized, agent_action
  ):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = min_normalized
    max_normalized_value = max_normalized

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    with self.assertRaises(ValueError):
      handler.setpoint_value(np.array(agent_action))

  def test_normalized_range_setpoint_value_shape_raises_error(self):
    min_native_value = 200
    max_native_value = 300

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value, max_native_value=max_native_value
    )

    with self.assertRaises(ValueError):
      handler.setpoint_value(np.array([1, 2]))

  def test_default_array_spec(self):
    min_native_value = 200
    max_native_value = 300
    name = 'action_1'
    expected_spec = specs.BoundedArraySpec(
        (), np.float32, minimum=-1, maximum=1, name=name
    )

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value, max_native_value=max_native_value
    )

    output_spec = handler.get_array_spec(name)

    self.assertEqual(output_spec, expected_spec)

  def test_normalized_range_array_spec(self):
    min_native_value = 200
    max_native_value = 300
    min_normalized_value = 0
    max_normalized_value = 1
    name = 'action_1'
    expected_spec = specs.BoundedArraySpec(
        (),
        np.float32,
        minimum=min_normalized_value,
        maximum=max_normalized_value,
        name=name,
    )

    handler = bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value=min_native_value,
        max_native_value=max_native_value,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
    )

    output_spec = handler.get_array_spec(name)

    self.assertEqual(output_spec, expected_spec)


if __name__ == '__main__':
  absltest.main()
