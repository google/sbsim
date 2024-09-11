"""Tests for agent_utils.

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
from smart_control.utils import agent_utils
import tensorflow as tf


class AgentUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ([-4.0, 0.5, 6.0, 2.0], 5, 120, 1.2, 10.0, -6.0),
      ([-0.2, 0.5, 0.0, 0.8], 1, 10, 0.01, 1.0, -1.0),
      ([-0.2], 25, 1, 0.3, 1.0, -1.0),
      ([200.0], 10, 10, 20.0, 800.0, 100.0),
      ([0.40] * 30, 30, 100, 0.1, 1.0, -1.0),
  )
  def test_create_random_walk_collect_script(
      self,
      policy_array,
      time_steps_per_random_step,
      policy_length,
      random_step_magnitude,
      upper_bound,
      lower_bound,
  ):
    fixed_policy = tf.constant(policy_array)
    num_action_dims = len(policy_array)

    random_walk_policy = agent_utils.create_random_walk_collect_script(
        fixed_policy=fixed_policy,
        time_steps_per_random_step=time_steps_per_random_step,
        policy_length=policy_length,
        random_step_magnitude=random_step_magnitude,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )

    with self.subTest(name='CheckPolicyLength'):
      self.assertLen(random_walk_policy, policy_length)
    last_step = None

    for step in random_walk_policy:
      # Check the policy length is correct.
      self.assertEqual(time_steps_per_random_step, step[0])
      if last_step is not None:
        # Check that the number of steps is correct.
        with self.subTest(name='CheckNumberSteps'):
          self.assertEqual(time_steps_per_random_step, step[0])

        with self.subTest(name='CheckStepDims'):
          self.assertEqual(step[1].shape, num_action_dims)

        # Check step sizes are no larger than random_step_magnitude.
        with self.subTest(name='CheckStepMagnitude'):
          dif = tf.math.subtract(last_step[1], step[1])
          abs_dif = tf.math.abs(dif)
          max_dif = np.max(abs_dif.numpy())
          # Need to add small tolerance  (1e-6) due to float32 imprecision.
          self.assertLessEqual(max_dif, random_step_magnitude + 1e-6)

      # Check that the results are always inside upper and lower limits.
      with self.subTest(name='CheckMaxValue'):
        max_val = np.max(step[1].numpy())
        self.assertLessEqual(max_val, upper_bound)
      with self.subTest(name='CheckMinValue'):
        min_val = np.min(step[1].numpy())
        self.assertGreaterEqual(min_val, lower_bound)

      last_step = step

  @parameterized.named_parameters(
      (
          'bad dims',
          [],
          0,
          120,
          0.1,
          0.2,
          -0.2,
          'Fixed policy must have at least one dim.',
      ),
      (
          'bad time steps',
          [0.9],
          0,
          120,
          1.2,
          10,
          -6.0,
          'Time steps per random steps must be int 1 or greater.',
      ),
      (
          'bad range',
          [0.2, -0.1, 2.001, 0.8],
          5,
          120,
          0.01,
          2.0,
          -2.0,
          'fixed_policy is above upper_bound or below lower_bound.',
      ),
      (
          'bad step magnitude',
          [0.04],
          5,
          120,
          0.1001,
          0.2,
          -0.2,
          (
              'Step magnitude must not be greater than 25% of range '
              'between upper and lower bounds.'
          ),
      ),
      (
          'zero magnitude step',
          [0.04, 0.05, 0.06, 0.08],
          5,
          120,
          0.0,
          0.2,
          -0.2,
          'Step magnitude must be greater than 0.',
      ),
      (
          'bad bounds',
          [0.4, 0.5, 0.6],
          5,
          120,
          0.2,
          0.2,
          0.2,
          'Upper bound must be greater than lower_bound.',
      ),
  )
  def test_create_random_walk_collect_error(
      self,
      fixed_policy_array,
      time_steps_per_random_step,
      policy_length,
      random_step_magnitude,
      upper_bound,
      lower_bound,
      err,
  ):
    fixed_policy = tf.constant(fixed_policy_array)

    with self.assertRaisesRegex(ValueError, err):
      _ = agent_utils.create_random_walk_collect_script(
          fixed_policy=fixed_policy,
          time_steps_per_random_step=time_steps_per_random_step,
          policy_length=policy_length,
          random_step_magnitude=random_step_magnitude,
          upper_bound=upper_bound,
          lower_bound=lower_bound,
      )


if __name__ == '__main__':
  absltest.main()
