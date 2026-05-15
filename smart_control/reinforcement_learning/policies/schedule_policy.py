"""Reinforcement learning schedule policies."""

import dataclasses
import enum
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

from smart_buildings.smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_buildings.smart_control.reinforcement_learning.utils.time_utils import to_dow
from smart_buildings.smart_control.reinforcement_learning.utils.time_utils import to_hod

logger = logging.getLogger(__name__)


# Device types that can be controlled
class DeviceType(enum.Enum):
  AC = 0
  HWS = 1


# Type aliases for clarity
SetpointName = str
SetpointValue = Union[float, int, bool]
ActionSequence = List[Tuple[DeviceType, SetpointName]]


@dataclasses.dataclass
class ScheduleEvent:
  """An event that sets a specific value at a specific time."""

  start_time: pd.Timedelta
  device: DeviceType
  setpoint_name: SetpointName
  setpoint_value: SetpointValue


# A schedule is a list of times and setpoints for devices
Schedule = List[ScheduleEvent]


def get_active_setpoint(
    schedule: Schedule,
    device: DeviceType,
    setpoint_name: SetpointName,
    timestamp: pd.Timedelta,
) -> SetpointValue:
  """Find the active setpoint value at a given time."""
  logger.debug('Getting active setpoint...')

  # Create a dictionary of {time: value} for the specific device and setpoint
  events = {
      event.start_time: event.setpoint_value
      for event in schedule
      if event.device == device and event.setpoint_name == setpoint_name
  }

  if not events:
    logger.exception('Events is None...')
    return None

  # Convert to Series for easier time-based lookup
  series = pd.Series(events)

  # Find events that happened at or before the timestamp
  prior_events = series.index[series.index <= timestamp]

  # If no prior events, wrap around and take the last event
  if prior_events.empty:
    return series.iloc[-1]
  else:
    return series.loc[prior_events[-1]]


class SchedulePolicy(tf_policy.TFPolicy):
  """Policy that selects actions based on time-dependent schedules."""

  def __init__(
      self,
      time_step_spec,
      action_spec: types.NestedTensorSpec,
      action_sequence: ActionSequence,
      weekday_schedule: Schedule,
      weekend_schedule: Schedule,
      dow_sin_index: int,
      dow_cos_index: int,
      hod_sin_index: int,
      hod_cos_index: int,
      action_normalizers: dict,  # pylint: disable=g-bare-generic # TODO: use a more specific type hint if possible
      local_start_time: pd.Timestamp,
      name: Optional[str] = None,
  ):
    self.weekday_schedule = weekday_schedule
    self.weekend_schedule = weekend_schedule
    self.dow_sin_index = dow_sin_index
    self.dow_cos_index = dow_cos_index
    self.hod_sin_index = hod_sin_index
    self.hod_cos_index = hod_cos_index
    self.action_sequence = action_sequence
    self.action_normalizers = action_normalizers
    self.local_start_time = local_start_time
    self.norm_mean = 0.0
    self.norm_std = 1.0

    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=(),
        info_spec=(),
        clip=False,
        observation_and_action_constraint_splitter=None,
        name=name,
    )

  def _normalize_actions(
      self, action_map: Dict[Tuple[DeviceType, SetpointName], SetpointValue]
  ) -> Dict:  # pylint: disable=g-bare-generic # TODO: use a more specific type hint if possible
    """Normalize action values using the provided normalizers."""
    normalized = {}
    for (device, setpoint_name), value in action_map.items():
      # Find the matching normalizer for this setpoint
      for normalizer_key, normalizer in self.action_normalizers.items():
        if normalizer_key.endswith(setpoint_name):
          normalized[(device, setpoint_name)] = normalizer.agent_value(value)
          break
    return normalized

  def _get_action_map(self, time_step) -> Dict:  # pylint: disable=g-bare-generic # TODO: use a more specific type hint if possible
    """Determine the appropriate actions based on time."""
    observation = time_step.observation

    # Denormalize the time signals
    # fmt: off
    # pylint: disable=line-too-long
    dow_sin = (observation[0][self.dow_sin_index] * self.norm_std) + self.norm_mean
    dow_cos = (observation[0][self.dow_cos_index] * self.norm_std) + self.norm_mean
    hod_sin = (observation[0][self.hod_sin_index] * self.norm_std) + self.norm_mean
    hod_cos = (observation[0][self.hod_cos_index] * self.norm_std) + self.norm_mean
    # pylint: enable=line-too-long
    # fmt: on

    # Convert to day of week and hour of day
    dow = to_dow(dow_sin, dow_cos)
    hod = to_hod(hod_sin, hod_cos)

    # Create timestamp
    timestamp = (
        pd.Timedelta(hod, unit='hour') + self.local_start_time.utcoffset()
    )

    # Use appropriate schedule based on day type
    schedule = self.weekday_schedule if dow < 5 else self.weekend_schedule

    # Get active setpoints for each device/setpoint pair
    return {
        (device, setpoint): get_active_setpoint(
            schedule, device, setpoint, timestamp
        )
        for device, setpoint in self.action_sequence
    }

  def _action(self, time_step, policy_state, seed):
    """Generate the policy action."""
    del seed, policy_state

    # Get and normalize actions
    action_map = self._get_action_map(time_step)
    normalized_map = self._normalize_actions(action_map)

    # Convert to array in the correct order
    action_array = np.array(
        [
            normalized_map[(device, setpoint)]
            for device, setpoint in self.action_sequence
        ],
        dtype=np.float32,
    )

    # Add batch dimension - this is the key fix
    action_array = np.expand_dims(action_array, axis=0)

    return policy_step.PolicyStep(tf.convert_to_tensor(action_array), (), ())


def create_baseline_schedule_policy(
    tf_env: tf_py_environment.TFPyEnvironment,
) -> SchedulePolicy:
  """Create baseline schedule policy.

  This is the baseline default policy that we use for benchmarking /
  initial data collection.

  Args:
    tf_env: The TFPyEnvironment to interact with.

  Returns:
    The schedule policy.
  """
  env = tf_env.pyenv.envs[0]

  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)

  hod_cos_index = env.field_names.index('hod_cos_000')
  hod_sin_index = env.field_names.index('hod_sin_000')
  dow_cos_index = env.field_names.index('dow_cos_000')
  dow_sin_index = env.field_names.index('dow_sin_000')
  # Note that temperatures are specified in Kelvin:
  weekday_schedule_events = [
      ScheduleEvent(
          pd.Timedelta(6, unit='hour'),
          DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          292.0,
      ),
      ScheduleEvent(
          pd.Timedelta(19, unit='hour'),
          DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          285.0,
      ),
      ScheduleEvent(
          pd.Timedelta(6, unit='hour'),
          DeviceType.HWS,
          'supply_water_setpoint',
          350.0,
      ),
      ScheduleEvent(
          pd.Timedelta(19, unit='hour'),
          DeviceType.HWS,
          'supply_water_setpoint',
          315.0,
      ),
  ]

  weekend_holiday_schedule_events = [
      ScheduleEvent(
          pd.Timedelta(6, unit='hour'),
          DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          285.0,
      ),
      ScheduleEvent(
          pd.Timedelta(19, unit='hour'),
          DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          285.0,
      ),
      ScheduleEvent(
          pd.Timedelta(6, unit='hour'),
          DeviceType.HWS,
          'supply_water_setpoint',
          315.0,
      ),
      ScheduleEvent(
          pd.Timedelta(19, unit='hour'),
          DeviceType.HWS,
          'supply_water_setpoint',
          315.0,
      ),
  ]

  local_start_time = env.current_simulation_timestamp.tz_convert(
      tz=DEFAULT_TIME_ZONE
  )

  baseline_schedule_policy = SchedulePolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      action_sequence=[
          (DeviceType.AC, 'supply_air_heating_temperature_setpoint'),
          (DeviceType.HWS, 'supply_water_setpoint'),
      ],
      weekday_schedule=weekday_schedule_events,
      weekend_schedule=weekend_holiday_schedule_events,
      action_normalizers=env.action_normalizers,
      hod_cos_index=hod_cos_index,
      hod_sin_index=hod_sin_index,
      dow_cos_index=dow_cos_index,
      dow_sin_index=dow_sin_index,
      local_start_time=local_start_time,
  )

  return baseline_schedule_policy
