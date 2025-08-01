"""Defines the utility functions used by the MPPI agent and world model.


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

import enum
import re
import numpy as np
from tf_agents.trajectories import policy_step
from smart_buildings.smart_control.utils import bounded_action_normalizer


class HistogramEstimationMethod(enum.Enum):
  """Method used to estimate a value from a histogram."""

  WEIGHTED_AVERAGE = 'weighted_average'
  MAX_PROBABILITY = 'max_probability'


def get_estimated_temp_from_histogram(
    observation_array: np.ndarray,
    all_field_names: list[str],
    sensor_base_name: str,  # e.g., "zone_air_temperature_sensor"
    method: HistogramEstimationMethod = HistogramEstimationMethod.WEIGHTED_AVERAGE,
) -> float | None:
  r"""Estimates a single temperature value from its histogram features in the observation.

  Args:
      observation_array: A numpy array containing the observation data.
      all_field_names: A list of strings representing the names of all fields in
        the observation.
      sensor_base_name: The base name of the sensor for which to estimate the
        temperature (e.g., "zone_air_temperature_sensor").
      method: The method to use for estimating the temperature. Can be either
        "weighted_average" or "max_probability".

  Returns:
      The estimated temperature value, or None if no valid histogram bins are
      found.

  The function expects histogram features in the observation array, with names
  that follow a specific pattern:
  `(.+_)?{sensor_base_name}_h_([0-9]+\\.?[0-9]*)`, where:
    - `(.+_)`: An optional prefix.
    - `{sensor_base_name}`: The base name of the sensor.
    - `_h_`: A literal string that separates the sensor name from the bin value.
    - `([0-9]+\\.?[0-9]*)`: The temperature value of the bin.

  """
  relevant_bins = []  # List of (temperature_value, probability_in_bin)

  pattern = re.compile(
      f'(.+_)?{re.escape(sensor_base_name)}_h_([0-9]+\\.?[0-9]*)'
  )

  for i, full_name in enumerate(all_field_names):
    match = pattern.fullmatch(full_name)
    if match:
      try:
        temp_str = match.group(2)  # The captured temperature part
        temp_val = float(temp_str)
        bin_probability = observation_array[i]
        if bin_probability > 0:  # Consider only bins with some presence
          relevant_bins.append({'temp': temp_val, 'prob': bin_probability})
      except ValueError:
        print(
            'Warning: Could not parse temperature from histogram bin name:'
            f' {full_name}'
        )
        continue

  if not relevant_bins:
    print(f'No active or parseable histogram bins found for {sensor_base_name}')
    return None

  if method == HistogramEstimationMethod.MAX_PROBABILITY:
    best_bin = max(relevant_bins, key=lambda x: x['prob'])
    return best_bin['temp']
  elif method == HistogramEstimationMethod.WEIGHTED_AVERAGE:
    weighted_sum = sum(b['temp'] * b['prob'] for b in relevant_bins)
    total_prob = sum(b['prob'] for b in relevant_bins)
    if total_prob > 0:
      return weighted_sum / total_prob
    else:
      print(
          f'Total probability for {sensor_base_name} bins is zero, cannot'
          ' compute weighted average.'
      )
      return None
  else:
    raise ValueError(f'Unknown method: {method}')


def apply_action(
    policy_step_instance: policy_step.PolicyStep,
    discrete_action_normalizers: dict[
        str, bounded_action_normalizer.BoundedActionNormalizer
    ],
    continuous_action_normalizers: dict[
        str, bounded_action_normalizer.BoundedActionNormalizer
    ],
    device_id: str,
    action_name: str,
    native_action_value: float,
) -> policy_step.PolicyStep:
  """Applies a single action to the policy step.

  This function modifies the provided `policy_step_instance` by setting the
  normalized value of a given action. It supports both discrete and continuous
  actions, and uses the provided normalizers to convert from native action
  values to agent values.

  Args:
      policy_step_instance: The PolicyStep object to modify.
      discrete_action_normalizers: A dictionary mapping device action names to
        discrete action normalizers.
      continuous_action_normalizers: A dictionary mapping device action names to
        continuous action normalizers.
      device_id: The ID of the device for which the action is applied.
      action_name: The name of the action to apply.
      native_action_value: The native value of the action to apply.

  Returns:
      The modified PolicyStep object.

  Raises:
      ValueError: If the device action name is not found in either the discrete
      or continuous action normalizers, or if an unknown method is provided.
  """
  device_action_name = device_id + '_' + action_name
  if device_action_name in discrete_action_normalizers:

    normalizer = discrete_action_normalizers[device_action_name]
    normalized_action_value = normalizer.agent_value(native_action_value)
    index = list(discrete_action_normalizers.keys()).index(device_action_name)
    policy_step_instance.action['discrete_action'][index] = (
        normalized_action_value
    )

  elif device_action_name in continuous_action_normalizers:
    normalizer = continuous_action_normalizers[device_action_name]
    normalized_action_value = normalizer.agent_value(native_action_value)
    index = list(continuous_action_normalizers.keys()).index(device_action_name)
    policy_step_instance.action['continuous_action'][index] = (
        normalized_action_value
    )

  else:
    raise ValueError(
        'Device action name %s not found in action normalizers.'
        % device_action_name
    )

  return policy_step_instance
