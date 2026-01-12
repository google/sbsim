"""Extension class for Environment that exposes hybrid actions to agent.

Intended for use with Hybrid Action - enabled agents that require both
continuous and discrete action spaces.
"""

import collections
from collections.abc import Sequence
from typing import Final

from absl import logging
import gin
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.typing import types

from smart_buildings.smart_control.environment import environment

_DISCRETE_ACTION: Final[str] = "discrete_action"
_CONTINUOUS_ACTION: Final[str] = "continuous_action"
_DISCRETE_ACTION_COMMAND: Final[str] = "supervisor_run_command"

HybridAction = dict[str, list[float]]


@gin.configurable
class HybridActionEnvironment(environment.Environment):
  """SB Environment that exposes discrete and continuous actions."""

  def _retrieve_field(
      self, device_id: environment.DeviceId, field_name: environment.FieldName
  ) -> environment.DeviceFieldId:
    field_id = environment.generate_field_id(
        device_id, field_name, self._id_map
    )
    if (device_id, field_name) not in self._id_map:
      self._id_map[(device_id, field_name)] = field_id
    return field_id

  def _get_action_spec_and_normalizers_from_device_action_tuples(
      self,
      action_config: environment.ActionConfig,
      device_action_tuples: Sequence[environment.DeviceActionTuple],
  ) -> tuple[types.ArraySpec, environment.ActionNormalizerMap, Sequence[str]]:
    """Applies the device_action_tuples to the action configurations."""
    action_normalizers = {}
    action_names = []

    discrete_action_names = []
    continuous_action_names = []

    logging.info(
        "Loading device-setpoint pairs from %d device_action_tuples.",
        len(device_action_tuples),
    )
    for device_action_tuple in device_action_tuples:
      device_id = environment.DeviceId(device_action_tuple[0])
      setpoint_name = environment.FieldName(device_action_tuple[1])

      # Get BaseActionNormalizer based on device and setpoint_name.
      action_normalizer = action_config.get_action_normalizer(setpoint_name)

      if not action_normalizer:
        raise ValueError("Missing a normalizer")

      field_id = self._retrieve_field(device_id, setpoint_name)

      if _DISCRETE_ACTION_COMMAND in setpoint_name:
        logging.info(
            "Device %s has a discrete action %s", device_id, setpoint_name
        )
        discrete_action_names.append(field_id)
      else:
        logging.info(
            "Device %s has a continuous action %s", device_id, setpoint_name
        )
        continuous_action_names.append(field_id)

      action_names.append(field_id)

      action_normalizers[field_id] = action_normalizer

    action_spec = {
        _CONTINUOUS_ACTION: array_spec.BoundedArraySpec(
            shape=(len(continuous_action_names),),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name=_CONTINUOUS_ACTION,
        ),
        _DISCRETE_ACTION: array_spec.BoundedArraySpec(
            shape=(len(discrete_action_names),),
            dtype=np.int32,
            minimum=0,
            maximum=1,
            name=_DISCRETE_ACTION,
        ),
    }
    logging.info(
        "The action_spec from device_action_tuples contains %d actions: %s.",
        len(action_names),
        ", ".join(action_names),
    )
    return action_spec, action_normalizers, action_names

  def _format_action(
      self, action: types.NestedArray, action_names: Sequence[str]
  ) -> types.NestedArray:  # to do: consider returning HybridAction type
    """Converts from hybrid to all real-valued actions."""
    if (
        not isinstance(action, dict)
        or _CONTINUOUS_ACTION not in action.keys()
        or _DISCRETE_ACTION not in action.keys()
    ):
      raise ValueError(
          "Hybrid Action Environment requires an action dict with continuous"
          " and discrete actions."
      )

    discrete_action = tf.reshape(
        action[_DISCRETE_ACTION], self._action_spec[_DISCRETE_ACTION].shape
    )
    discrete_dequeue = collections.deque(discrete_action)
    continuous_dequeue = collections.deque(action[_CONTINUOUS_ACTION])

    if len(discrete_dequeue) + len(continuous_dequeue) != len(action_names):
      raise ValueError(
          f"The number of discrete actions was {len(discrete_dequeue)} and"
          " continuous actions was {len(continuous_dequeue)}, did not add"
          " up to the expected number of actions: {len(action_names)}"
      )

    merged_actions = []
    # Here we take a dictionary with discrete actions as integers and
    # continuous actions as floats, and construct a list of floats in
    # the order provided by action names.
    # The discrete and continuous actions are already ordered, but
    # they need to be merged into a single float list.
    # Only discrete actions with _DISCRETE_ACTION_COMMMAND in the name
    # are recognized as discrete.
    for action_name in action_names:
      if _DISCRETE_ACTION_COMMAND in action_name:
        discrete_action_value = discrete_dequeue.popleft()

        # The convention for the agent is 1 on and 0 off, but in the
        # base building is (0.0, 1.0) on and [0.0, -1.0) off.
        if discrete_action_value == 1:
          merged_actions.append(1.0)
        elif discrete_action_value == 0:
          merged_actions.append(-1.0)
        else:
          raise NotImplementedError(
              "Only [0, 1] discrete actions are supported now, but received a"
              f" {discrete_action_value} for field {action_name}."
          )

      else:
        continuous_action_value = continuous_dequeue.popleft()
        merged_actions.append(continuous_action_value)

    if len(merged_actions) != len(action_names):
      raise ValueError(
          "The number of merged actions did not match the number of expected"
          " actions, which may indicate that there is a discrete action that is"
          " mislabeled."
      )

    return merged_actions

  @property
  def action_fields_df(self) -> pd.DataFrame:
    df = super().action_fields_df
    # override action_type column, with awareness of discrete actions:
    df["action_type"] = df["setpoint_name"].apply(
        lambda name: (
            "DISCRETE"
            if _DISCRETE_ACTION_COMMAND in name
            else "CONTINUOUS"
        )
    )
    return df
