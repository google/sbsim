import numpy as np
import pandas as pd
import enum
import tensorflow as tf
from typing import Union, Optional, cast
from dataclasses import dataclass
from tf_agents.trajectories import policy_step
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

# We're concerned with controlling Heatpumps/ACs and Hot Water Systems (HWS).
class DeviceType(enum.Enum):
    AC = 0
    HWS = 1


SetpointName = str  # Identify the setpoint
SetpointValue = Union[float, int, bool]


@dataclass
class ScheduleEvent:
    start_time: pd.Timedelta
    device: DeviceType
    setpoint_name: SetpointName
    setpoint_value: SetpointValue


# A schedule is a list of times and setpoints for a device.
Schedule = list[ScheduleEvent]
ActionSequence = list[tuple[DeviceType, SetpointName]]


def to_rad(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to radians to extract the time."""

    if sin_theta >= 0 and cos_theta >= 0:
        return np.arccos(cos_theta)
    elif sin_theta >= 0 and cos_theta < 0:
        return np.pi - np.arcsin(sin_theta)
    elif sin_theta < 0 and cos_theta < 0:
        return np.pi - np.arcsin(sin_theta)
    else:
        return 2 * np.pi - np.arccos(cos_theta)


def to_dow(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to days to extract day of week."""
    theta = to_rad(sin_theta, cos_theta)
    return np.floor(7 * theta / 2 / np.pi)


def to_hod(sin_theta: float, cos_theta: float) -> float:
    """Converts a sin and cos theta to hours to extract hour of day."""
    theta = to_rad(sin_theta, cos_theta)
    return np.floor(24 * theta / 2 / np.pi)


def find_schedule_action(
    schedule: Schedule,
    device: DeviceType,
    setpoint_name: SetpointName,
    timestamp: pd.Timedelta,
) -> SetpointValue:
    """Finds the action for a schedule event for a time and schedule."""

    # Get all the schedule events for the device and the setpoint, and turn it
    # into a series.
    device_schedule_dict = {}
    for schedule_event in schedule:
        if (
            schedule_event.device == device
            and schedule_event.setpoint_name == setpoint_name
        ):
            device_schedule_dict[schedule_event.start_time] = (
                schedule_event.setpoint_value
            )
    device_schedule = pd.Series(device_schedule_dict)

    # Get the indexes of the schedule events that fall before the timestamp.

    device_schedule_indexes = device_schedule.index[
        device_schedule.index <= timestamp
    ]

    # If are no events preceedding the time, then choose the last
    # (assuming it wraps around).
    if device_schedule_indexes.empty:
        return device_schedule.loc[device_schedule.index[-1]]
    else:
        return device_schedule.loc[device_schedule_indexes[-1]]

# @title Define a schedule policy
class SchedulePolicy(tf_policy.TFPolicy):
    """TF Policy implementation of the Schedule policy."""

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        action_sequence: ActionSequence,
        weekday_schedule_events: Schedule,
        weekend_holiday_schedule_events: Schedule,
        dow_sin_index: int,
        dow_cos_index: int,
        hod_sin_index: int,
        hod_cos_index: int,
        action_normalizers,
        local_start_time: str = pd.Timestamp,
        policy_state_spec: types.NestedTensorSpec = (),
        info_spec: types.NestedTensorSpec = (),
        name: Optional[str] = None,
    ):
        self.weekday_schedule_events = weekday_schedule_events
        self.weekend_holiday_schedule_events = weekend_holiday_schedule_events
        self.dow_sin_index = dow_sin_index
        self.dow_cos_index = dow_cos_index
        self.hod_sin_index = hod_sin_index
        self.hod_cos_index = hod_cos_index
        self.action_sequence = action_sequence
        self.action_normalizers = action_normalizers
        self.local_start_time = local_start_time
        self.norm_mean = 0.0
        self.norm_std = 1.0

        policy_state_spec = ()

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            clip=False,
            observation_and_action_constraint_splitter=None,
            name=name,
        )

    def _normalize_action_map(
        self, action_map: dict[tuple[DeviceType, SetpointName], SetpointValue]
    ) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        normalized_action_map = {}

        for k, v in action_map.items():
            for normalizer_k, normalizer in self.action_normalizers.items():
                if normalizer_k.endswith(k[1]):

                    normed_v = normalizer.agent_value(v)
                    normalized_action_map[k] = normed_v

        return normalized_action_map

    def _get_action(
        self, time_step
    ) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        observation = time_step.observation
        action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)
        dow_sin = (observation[self.dow_sin_index] * self.norm_std) + self.norm_mean
        dow_cos = (observation[self.dow_cos_index] * self.norm_std) + self.norm_mean
        hod_sin = (observation[self.hod_sin_index] * self.norm_std) + self.norm_mean
        hod_cos = (observation[self.hod_cos_index] * self.norm_std) + self.norm_mean

        dow = to_dow(dow_sin, dow_cos)
        hod = to_hod(hod_sin, hod_cos)

        timestamp = (
            pd.Timedelta(hod, unit='hour') + self.local_start_time.utcoffset()
        )

        if dow < 5:  # weekday

            action_map = {
                (tup[0], tup[1]): find_schedule_action(
                    self.weekday_schedule_events, tup[0], tup[1], timestamp
                )
                for tup in self.action_sequence
            }

            return action_map

        else:  # Weekend

            action_map = {
                (tup[0], tup[1]): find_schedule_action(
                    self.weekend_holiday_schedule_events, tup[0], tup[1], timestamp
                )
                for tup in self.action_sequence
            }

            return action_map

    def _action(self, time_step, policy_state, seed):
        del seed
        action_map = self._get_action(time_step)
        normalized_action_map = self._normalize_action_map(action_map)

        action = np.array(
            [
                normalized_action_map[device_setpoint]
                for device_setpoint in self.action_sequence
            ],
            dtype=np.float32,
        )

        t_action = tf.convert_to_tensor(action)

        return policy_step.PolicyStep(t_action, (), ())


def find_fixed_action(
    schedule: Schedule,
    device: DeviceType,
    setpoint_name: SetpointName
) -> SetpointValue:
    """Finds the action for a schedule event for a time and schedule."""

    # Get all the schedule events for the device and the setpoint, and turn it
    # into a series.
    setpoint_dict = {}
    for schedule_event in schedule:
        if (schedule_event.device == device and schedule_event.setpoint_name == setpoint_name):
            setpoint_value = schedule_event.setpoint_value

    return setpoint_value


class FixedActionPolicy(tf_policy.TFPolicy):
    """TF Policy implementation of the Schedule policy."""

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        action_sequence: ActionSequence,
        schedule_events: Schedule,
        action_normalizers,
        policy_state_spec: types.NestedTensorSpec = (),
        info_spec: types.NestedTensorSpec = (),
        name: Optional[str] = None,
    ):
        self.schedule_events = schedule_events
        self.action_sequence = action_sequence
        self.action_normalizers = action_normalizers
        self.norm_mean = 0.0
        self.norm_std = 1.0

        policy_state_spec = ()

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            clip=False,
            observation_and_action_constraint_splitter=None,
            name=name,
        )

    def _normalize_action_map(
        self, action_map: dict[tuple[DeviceType, SetpointName], SetpointValue]
    ) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        normalized_action_map = {}

        for k, v in action_map.items():
            for normalizer_k, normalizer in self.action_normalizers.items():
                if normalizer_k.endswith(k[1]):

                    normed_v = normalizer.agent_value(v)
                    normalized_action_map[k] = normed_v

        return normalized_action_map

    def _get_action(
        self, time_step
    ) -> dict[tuple[DeviceType, SetpointName], SetpointValue]:

        action_map = {
            (tup[0], tup[1]): find_fixed_action(
                self.schedule_events, tup[0], tup[1]
            )
            for tup in self.action_sequence
        }

        return action_map

    def _action(self, time_step, policy_state, seed):
        del seed
        action_map = self._get_action(time_step)
        normalized_action_map = self._normalize_action_map(action_map)

        action = np.array(
            [
                normalized_action_map[device_setpoint]
                for device_setpoint in self.action_sequence
            ],
            dtype=np.float32,
        )

        t_action = tf.convert_to_tensor(action)

        return policy_step.PolicyStep(t_action, (), ())
