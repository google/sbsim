"""Schedule policy agent.

This agent determines its actions based on the building's operational
schedule. Based on the current date and time, if the building is operational,
the agent will turn on all devices and use the environment's default setpoint
values. Otherwise, when the building is not operational, the agent will turn off
all devices.

This agent requires a hybrid action environment, because it needs a mechanism
for turning devices on and off.
"""

from typing import Final

import numpy as np

from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import default_agent
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.schema import output_schema
from smart_buildings.smart_control.llm.utils import schedule_tool as schedule_lib
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import serialization


NATIVE_ON: Final[float] = 1.0
NATIVE_OFF: Final[float] = 0.0


class SchedulePolicyAgent(default_agent.DefaultPolicyAgent):
  """An agent that determines its actions based on the building's schedule.

  Based on the current date and time, if the building is operational, the agent
  will turn on all devices and use the environment's default setpoint values.
  Otherwise, when the building is not operational, the agent will turn off all
  devices.

  Because it is possible (but not common) for an environment's default values to
  specify a device should be off, if you want to preserve that behavior and
  prevent those devices from being turned on during operational hours, set the
  `override_discrete_defaults` to `False`, and the agent will respect those
  default values.

  This agent is to be used in conjunction with a hybrid action environment, so
  it has a mechanism for turning devices ON or OFF.

  Attributes:
    schedule_tool: The BuildingScheduleTool instance used to determine the
      building's operational schedule.
    override_discrete_defaults: Whether to override the environment's default
      values for discrete actions (e.g., turning devices ON/OFF) during
      operational hours. If True, discrete devices will be turned ON during
      operational hours, even if the default value is OFF. If False, the
      default discrete values are respected.
  """

  def __init__(
      self,
      *,
      env: hybrid_action_environment.HybridActionEnvironment,
      clip: bool = True,
      override_discrete_defaults: bool = True,
      schedule_tool: schedule_lib.BuildingScheduleTool | None = None,
  ):
    """Initializes the instance.

    Args:
      env: The hybrid action environment the agent will interact with.
      clip: Whether to clip setpoint values to the bounds of the valid range. If
        `False`, raises `GuardrailsExceededError` if setpoint values are out of
        range. Otherwise, clips the setpoint values to the valid range, and logs
        a record of the error. Defaults to `True`.
      override_discrete_defaults: Whether to override the default policy values
        for discrete actions during operational hours. By default, the agent
        will turn on all devices during operational hours, potentially
        overriding any default values that specify a device should be off. If
        you have default values that specify a device should be off during
        operational hours, set this option to `False` and the agent will respect
        those defaults.
      schedule_tool: Optionally provide a BuildingScheduleTool instance.
        Otherwise, a schedule tool will be constructed using the agent's
        environment and default schedule tool arguments.
    """
    super().__init__(env=env, clip=clip)
    self.schedule_tool = schedule_tool or schedule_lib.BuildingScheduleTool(
        env=env,
    )
    self.override_discrete_defaults = override_discrete_defaults

  @property
  def json_metadata(self) -> serialization.SerializableData:
    """Info to write into a JSON file. Needs to be serializable."""
    return super().json_metadata | {
        "override_discrete_defaults": self.override_discrete_defaults,
        "schedule_policy": self.schedule_tool.json_metadata,
    }

  @property
  def building_operational_mode(self) -> schedule_lib.BuildingOperationalMode:
    """The building's operational mode."""
    return self.schedule_tool.building_operational_mode

  @property
  def building_is_operational(self) -> bool:
    """Whether the building is operational."""
    return self.schedule_tool.building_is_operational

  @property
  def scheduled_justification(self) -> str:
    return f"Scheduled action ({self.building_operational_mode.value.upper()})"

  @property
  def scheduled_setpoint_justification(self) -> str:
    return f"Scheduled value ({self.building_operational_mode.value.upper()})"

  def get_scheduled_native_value(
      self, setpoint_name: str, native_value: float
  ) -> float:
    """Determines the scheduled native value for a given setpoint.

    This method will flip the value of discrete actions to ON or OFF, depending
    on whether the building is operational or not.

    Because it is possible (but not common) for an environment's default values
    to specify a device should be off, if you want to preserve that behavior and
    prevent those devices from being turned on during operational hours, set the
    `override_discrete_defaults` to `False`, and the agent will respect those
    default values.

    Args:
      setpoint_name: The name of a given setpoint.
      native_value: The native action value for the given setpoint.

    Returns:
      The scheduled native action value for the setpoint.
    """
    if not hybrid_action_environment.is_discrete_action(setpoint_name):
      return native_value

    if self.building_is_operational:
      return NATIVE_ON if self.override_discrete_defaults else native_value
    return NATIVE_OFF

  def get_scheduled_action_context(self) -> action_context.ActionContext:
    """Gets an action context based on the building's operational schedule.

    This action context uses the environment's default policy values as a base,
    but ensures devices are turned off during non-operational hours, and on
    during operational hours.

    Returns:
      An action context representing the scheduled action.
    """
    setpoints = []
    for action_name, normalized_value in zip(
        self.env.action_names, self.env.default_policy_values, strict=True
    ):
      device_id, setpoint_name = self.env.id_map.inv[action_name]

      normalizer = self.env.action_normalizers.get(setpoint_name)
      if normalizer is None:
        raise ValueError(f"No normalizer found for setpoint: {setpoint_name}")

      native_value = normalizer.setpoint_value(np.array(normalized_value))
      scheduled_native_value = self.get_scheduled_native_value(
          setpoint_name=setpoint_name, native_value=native_value
      )

      setpoints.append(
          output_schema.DeviceSetpoint(
              device_id=device_id,
              setpoint_name=setpoint_name,
              setpoint_value=scheduled_native_value,
              justification=self.scheduled_setpoint_justification,
          )
      )

    return self.action_context_class(
        env=self.env,
        clip=self._clip,
        timestamp=str(self.env.current_local_timestamp),
        justification=self.scheduled_justification,
        validity_interval=self.env.time_step_mins,
        setpoints=setpoints,
    )

  def get_action_context(
      self,
      observation_response: building_pb2.ObservationResponse | None = None,
      reward_info: reward_pb2.RewardInfo | None = None,
  ) -> action_context.ActionContext:
    """Returns the action context for the environment."""
    del observation_response, reward_info  # Unused in this implementation.
    return self.get_scheduled_action_context()
