"""Utilities for training Smart Building Reinforcement Learning agents.

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

from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf

# PolicyStep = number of agent steps to hold the policy constant.
# Using a custom type of Tuple[int, Tensor] since it provides a
# named type while retaining the structure and form defined by
# TF-Agants.
PolicyStep = Tuple[int, tf.Tensor]
# Policy = a list of policy steps in the episode.
Policy = Sequence[PolicyStep]


def create_random_walk_collect_script(
    fixed_policy: tf.Tensor,
    time_steps_per_random_step: int,
    policy_length: int,
    random_step_magnitude: float,
    upper_bound: float = 1,
    lower_bound: float = -1,
) -> Policy:
  """Returns a scripted random walk policy for collects on agent training.

  A scripted policy is used by TF Agents to script actions, regardless of the
  current state. The format is a list of transition pairs, where the first
  element is an integer that specifies how many steps the transition should
  hold for (1 or more), and the second is a Tensor of actions to apply,
  which are bounded. The policy length are the number of transitions provided.

  A random walk script is a a scripted policy that takes a random step
  of size random_step_magnitude, independently for each action dimension,
  which is held constant for time_steps_per_random_step.

  The random walk is a a technique to enable the agent to explore action space.

  Args:
    fixed_policy: A Tensor of initial values for the random walk.
    time_steps_per_random_step: Duration in time steps for each random step.
    policy_length: Number of random policy changes to return.
    random_step_magnitude: Fixed magnitude perturbation for each random step.
    upper_bound: Upper limit of the random walk.
    lower_bound: Lower limit of the random walk.

  Returns:
    Random walk policy as list of (timesteps, action values tensor).

  Raises:
    ValueError if fixed_policy is above upper_bound or below lower_bound.
    ValueError if upper_bound <= lower_bound.
    ValueError if random_step_magnitude is not positive.
    ValueError if random_step_magnitude > 25% of upper_bound - lower_bound.
    ValueError if time steps per random step is less than 1.
    ValueError if fixed policy is empty, 0 dim.
  """
  if fixed_policy.shape[0] < 1:
    raise ValueError('Fixed policy must have at least one dim.')

  if upper_bound <= lower_bound:
    raise ValueError('Upper bound must be greater than lower_bound.')

  if (
      np.max(fixed_policy.numpy()) > upper_bound
      or np.min(fixed_policy.numpy()) < lower_bound
  ):
    raise ValueError('fixed_policy is above upper_bound or below lower_bound.')

  if random_step_magnitude <= 0.0:
    raise ValueError('Step magnitude must be greater than 0.')

  if random_step_magnitude > 0.25 * (upper_bound - lower_bound):
    raise ValueError(
        'Step magnitude must not be greater than 25% '
        'of range between upper and lower bounds.'
    )

  if time_steps_per_random_step < 1:
    raise ValueError('Time steps per random steps must be int 1 or greater.')

  action_dimensionality = fixed_policy.shape[0]
  upper = tf.constant([upper_bound] * action_dimensionality)
  lower = tf.constant([lower_bound] * action_dimensionality)
  random_walk_policy = [(time_steps_per_random_step, fixed_policy)]
  while len(random_walk_policy) < policy_length:
    random_step = tf.constant(
        np.random.choice([-1.0, 1.0], size=action_dimensionality, p=[0.5, 0.5])
        * random_step_magnitude,
        dtype=np.float32,
    )

    last_vals = random_walk_policy[-1][1]
    next_vals = tf.math.add(last_vals, random_step)

    next_values_clipped = tf.clip_by_value(
        next_vals, clip_value_min=lower, clip_value_max=upper
    )

    random_walk_policy.append((time_steps_per_random_step, next_values_clipped))

  return random_walk_policy
