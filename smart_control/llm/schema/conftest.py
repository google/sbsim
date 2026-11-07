"""Test helpers for LLM prompts and output schema models.

Contains objects for representing the LLM's response in string format, as well
as the corresponding Pydantic models parsed from those strings.

Provides actions for both the continuous and hybrid action environments.
"""

from collections.abc import Sequence
import json
import re
import textwrap
from typing import Any

from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.schema import output_schema

DISCRETE_ACTION_COMMAND = hybrid_action_environment.DISCRETE_ACTION_COMMAND

DEFAULT_VALIDITY_INTERVALS = output_schema.DEFAULT_VALIDITY_INTERVALS

EXAMPLE_TIMESTAMP = "2025-01-01 12:00:00"
EXAMPLE_JUSTIFICATION = "These are my overall goals."
EXAMPLE_DEVICE_JUSTIFICATION = "The reason for choosing this setpoint value."


def parse_instructions_schema(instructions: str) -> dict[str, Any] | None:
  """Parses a string containing a Pydantic schema, returns the schema data."""
  instructions = textwrap.dedent(instructions).strip()
  match = re.search(r"```\n({.*})\n```", instructions, re.DOTALL)
  if match:
    json_string = match.group(1)
    try:
      schema = json.loads(json_string)
      return schema
    except json.JSONDecodeError:
      return None
  return None


#
# DEVICE SETPOINTS
#


def create_supply_air_heating_temperature_setpoint(
    device_id: str = "air_handler_0",
    setpoint_value: float = 285.0,
    justification: str = "To cool the air.",
) -> output_schema.DeviceSetpoint:
  """Creates a supply air heating temperature setpoint for a specific device."""
  return output_schema.DeviceSetpoint(
      device_id=device_id,
      setpoint_name="supply_air_heating_temperature_setpoint",
      setpoint_value=setpoint_value,
      justification=justification,
  )


def create_supply_water_setpoint(
    device_id: str = "boiler_0",
    setpoint_value: float = 310.0,
    justification: str = "To heat the water.",
) -> output_schema.DeviceSetpoint:
  """Creates a supply water temperature setpoint for a specific device."""
  return output_schema.DeviceSetpoint(
      device_id=device_id,
      setpoint_name="supply_water_setpoint",
      setpoint_value=setpoint_value,
      justification=justification,
  )


def create_supervisor_run_command_setpoint(
    device_id: str = "air_handler_0",
    setpoint_value: float = 1,
    justification: str = "To turn the device on.",
) -> output_schema.DeviceSetpoint:
  """Creates a supervisor run command setpoint for a specific device."""
  return output_schema.DeviceSetpoint(
      device_id=device_id,
      setpoint_name=DISCRETE_ACTION_COMMAND,
      setpoint_value=setpoint_value,
      justification=justification,
  )


#
# ACTIONS (CONTINUOUS)
#


def create_action_response(
    ahu_1_supply_air_temp: float = 285.0,  # -1.0 (bottom of range)
    ahu_2_supply_air_temp: float = 295.0,  # 1.0 (top of range)
    hws_supply_water_temp: float = 330.0,  # 0.0 (middle of range)
    empty_setpoints: bool = False,
    missing_setpoint: bool = False,
    missing_field: bool = False,
    validity_interval: int = 60,
) -> str:
  """Creates an action response for the continuous action environment.

  Provides convenience arguments for creating invalid responses. Only one of
  these arguments (empty_setpoints, missing_setpoint, missing_field) should be
  set to True at a time.

  Args:
    ahu_1_supply_air_temp: The setpoint temp in Kelvin for AHU-1.
    ahu_2_supply_air_temp: The setpoint temp in Kelvin for AHU-2.
    hws_supply_water_temp: The setpoint temp in Kelvin for HWS.
    empty_setpoints: Whether to remove all setpoints from the response, to make
      it invalid.
    missing_setpoint: Whether to remove a setpoint from the response, to make it
      invalid (from the environment's perspective only).
    missing_field: Whether to remove a field from a setpoint, to make it
      invalid.
    validity_interval: The selected validity interval, in minutes.

  Returns:
    The action response as a JSON-formatted string.
  """

  action_data = {
      "timestamp": EXAMPLE_TIMESTAMP,
      "justification": EXAMPLE_JUSTIFICATION,
      "validity_interval": validity_interval,
      "setpoints": [
          {
              "device_id": "air_handler_1",
              "setpoint_name": "supply_air_heating_temperature_setpoint",
              "setpoint_value": ahu_1_supply_air_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "boiler_1",
              "setpoint_name": "supply_water_setpoint",
              "setpoint_value": hws_supply_water_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "air_handler_2",
              "setpoint_name": "supply_air_heating_temperature_setpoint",
              "setpoint_value": ahu_2_supply_air_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
      ],
  }

  if sum([empty_setpoints, missing_setpoint, missing_field]) > 1:
    raise ValueError(
        "Only one of empty_setpoints, missing_setpoint, or missing_field can be"
        " set to True at a time."
    )

  if missing_field:
    del action_data["setpoints"][0]["justification"]

  if missing_setpoint:
    del action_data["setpoints"][0]

  if empty_setpoints:
    action_data["setpoints"] = []

  # Convert data to a JSON-formatted string (to resemble the LLM's response):
  return textwrap.dedent(json.dumps(action_data, indent=2))


def create_action() -> output_schema.SetpointsAction:
  return output_schema.SetpointsAction(
      timestamp=EXAMPLE_TIMESTAMP,
      justification=EXAMPLE_JUSTIFICATION,
      validity_interval=60,
      setpoints=[
          create_supply_air_heating_temperature_setpoint("air_handler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_2"),
          create_supply_water_setpoint("boiler_1"),
      ],
  )


def create_action_context(
    env: environment.Environment | None = None,
    action: output_schema.SetpointsAction | None = None,
) -> action_context.ActionContext:
  """Creates an action context for the continuous action environment."""
  env = env or env_conftest.create_environment(layout=env_conftest.DEMO_LAYOUT)
  action = action or create_action()
  return action_context.ActionContext(env=env, **action.model_dump())


def create_action_with_custom_intervals(
    validity_intervals: Sequence[int] = DEFAULT_VALIDITY_INTERVALS,
    selected_interval: int = 60,
) -> output_schema.SetpointsAction:
  """Creates a SetpointsAction with custom validity intervals.

  Args:
    validity_intervals: The list of possible validity intervals in minutes.
    selected_interval: The selected validity interval in minutes.

  Returns:
    A SetpointsAction object with custom validity intervals.
  """
  model_class = output_schema.create_action_model(
      custom_intervals=validity_intervals
  )

  return model_class(
      timestamp=EXAMPLE_TIMESTAMP,
      justification=EXAMPLE_JUSTIFICATION,
      validity_interval=selected_interval,
      setpoints=[
          create_supply_air_heating_temperature_setpoint("air_handler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_2"),
          create_supply_water_setpoint("boiler_1"),
      ],
  )


#
# ACTIONS (HYBRID)
#


def create_hybrid_action_response(
    ahu_1_supply_air_temp: float = 285.0,  # -1.0 (bottom of range)
    ahu_2_supply_air_temp: float = 295.0,  # 1.0 (top of range)
    hws_supply_water_temp: float = 330.0,  # 0.0 (middle of range)
    ahu_1_run_command: int = 1,  # ON
    ahu_2_run_command: int = 1,  # ON
    hws_run_command: int = 1,  # ON
    empty_setpoints: bool = False,
    missing_setpoint: bool = False,
    missing_field: bool = False,
    validity_interval: int = 60,
) -> str:
  """Creates an action response for the hybrid action environment.

  Provides convenience arguments for creating invalid responses. Only one of
  these arguments (empty_setpoints, missing_setpoint, missing_field) should be
  set to True at a time.

  Args:
    ahu_1_supply_air_temp: The setpoint temp in Kelvin for AHU-1.
    ahu_2_supply_air_temp: The setpoint temp in Kelvin for AHU-2.
    hws_supply_water_temp: The setpoint temp in Kelvin for HWS.
    ahu_1_run_command: The run command for AHU-1.
    ahu_2_run_command: The run command for AHU-2.
    hws_run_command: The run command for HWS.
    empty_setpoints: Whether to remove all setpoints from the response, to make
      it invalid.
    missing_setpoint: Whether to remove a setpoint from the response, to make it
      invalid (from the environment's perspective only).
    missing_field: Whether to remove a field from a setpoint, to make it
      invalid.
    validity_interval: The selected validity interval, in minutes.

  Returns:
    The action response as a JSON-formatted string.
  """

  action_data = {
      "timestamp": EXAMPLE_TIMESTAMP,
      "validity_interval": validity_interval,
      "justification": EXAMPLE_JUSTIFICATION,
      "setpoints": [
          {
              "device_id": "air_handler_1",
              "setpoint_name": DISCRETE_ACTION_COMMAND,
              "setpoint_value": ahu_1_run_command,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "air_handler_2",
              "setpoint_name": DISCRETE_ACTION_COMMAND,
              "setpoint_value": ahu_2_run_command,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "boiler_1",
              "setpoint_name": DISCRETE_ACTION_COMMAND,
              "setpoint_value": hws_run_command,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "air_handler_1",
              "setpoint_name": "supply_air_heating_temperature_setpoint",
              "setpoint_value": ahu_1_supply_air_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "air_handler_2",
              "setpoint_name": "supply_air_heating_temperature_setpoint",
              "setpoint_value": ahu_2_supply_air_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
          {
              "device_id": "boiler_1",
              "setpoint_name": "supply_water_setpoint",
              "setpoint_value": hws_supply_water_temp,
              "justification": EXAMPLE_DEVICE_JUSTIFICATION,
          },
      ],
  }

  if sum([empty_setpoints, missing_setpoint, missing_field]) > 1:
    raise ValueError(
        "Only one of empty_setpoints, missing_setpoint, or missing_field can be"
        " set to True at a time."
    )

  if missing_setpoint:
    del action_data["setpoints"][0]

  if missing_field:
    del action_data["setpoints"][0]["justification"]

  if empty_setpoints:
    action_data["setpoints"] = []

  # Convert data to a JSON-formatted string (to resemble the LLM's response):
  return textwrap.dedent(json.dumps(action_data, indent=2))


def create_hybrid_action() -> output_schema.SetpointsAction:
  return output_schema.SetpointsAction(
      timestamp=EXAMPLE_TIMESTAMP,
      justification=EXAMPLE_JUSTIFICATION,
      validity_interval=60,
      setpoints=[
          create_supervisor_run_command_setpoint("air_handler_1"),
          create_supervisor_run_command_setpoint("air_handler_2"),
          create_supervisor_run_command_setpoint("boiler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_2"),
          create_supply_water_setpoint("boiler_1"),
      ],
  )


def create_hybrid_action_context(
    env: hybrid_action_environment.HybridActionEnvironment | None = None,
    action: output_schema.SetpointsAction | None = None,
) -> action_context.HybridActionContext:
  """Creates an action context for the hybrid action environment."""
  env = env or env_conftest.create_hybrid_action_environment(
      layout=env_conftest.DEMO_LAYOUT
  )
  action = action or create_hybrid_action()
  return action_context.HybridActionContext(env=env, **action.model_dump())


def create_hybrid_action_with_custom_intervals(
    validity_intervals: Sequence[int] = DEFAULT_VALIDITY_INTERVALS,
    selected_interval: int = 60,
) -> output_schema.SetpointsAction:
  """Creates a SetpointsAction with hybrid action and custom validity intervals.

  Args:
    validity_intervals: The list of possible validity intervals in minutes.
    selected_interval: The selected validity interval in minutes.

  Returns:
    A SetpointsAction object with custom validity intervals.
  """
  model_class = output_schema.create_action_model(
      custom_intervals=validity_intervals
  )

  return model_class(
      timestamp=EXAMPLE_TIMESTAMP,
      justification=EXAMPLE_JUSTIFICATION,
      validity_interval=selected_interval,
      setpoints=[
          create_supervisor_run_command_setpoint("air_handler_1"),
          create_supervisor_run_command_setpoint("air_handler_2"),
          create_supervisor_run_command_setpoint("boiler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_1"),
          create_supply_air_heating_temperature_setpoint("air_handler_2"),
          create_supply_water_setpoint("boiler_1"),
      ],
  )
