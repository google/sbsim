"""Model classes for defining the structure of LLM responses.

Includes models for the individual device setpoints, as well as action
models that contain any number of setpoints, and represent the full response
from an LLM, including other context such as the overall goals of the action.

For flexibility, we define a base action model which uses a default
set of validity interval options, however we also provide a method for
creating a model that uses a custom set of validity interval options.

The action model class itself is used to provide LLM formatting instructions,
which are derived from the model class schema.

When the LLM responds, its response can be used to initialize the action model
class, and can be validated using the model class validator.
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias

import pydantic

from smart_buildings.smart_control.utils import serialization

Field = pydantic.Field

DEFAULT_VALIDITY_INTERVALS = (5, 10, 15, 20, 30, 45, 60, 75, 90, 120)
DefaultValidityIntervalOptions: TypeAlias = Literal[*DEFAULT_VALIDITY_INTERVALS]  # pytype: disable=invalid-annotation  # pydantic needs it this way

TIMESTAMP_DESCRIPTION = (
    "The time the action is taken, formatted as 'YYYY-MM-DD HH:MM:SS', assumed"
    " to be in the building's local timezone."
)

JUSTIFICATION_DESCRIPTION = (
    "The overall reason for taking this action. Includes a brief description"
    " of why the action is justified, as well as the desired outcome of the"
    " action as a whole."
)

VALIDITY_INTERVAL_DESCRIPTION = (
    "The number of minutes the setpoints should remain in effect before"
    " prompting for a new action."
)


class DeviceSetpoint(pydantic.BaseModel):
  """A single device setpoint.

  A device is uniquely identified by a composite key consisting of the device
  identifier and the setpoint name.

  Attributes:
    device_id: The unique identifier of the device (e.g. 'boiler-123-xyz').
    setpoint_name: The name of the setpoint (e.g. 'supply_water_temperature').
    setpoint_value: The requested value to be set (e.g. 120.0).
    justification: The reason for choosing this specific device setting.
  """

  device_id: str = Field(
      description="The unique identifier of the device."
  )

  setpoint_name: str = Field(
      description="The name of the setpoint."
  )

  setpoint_value: float = Field(description="The requested value to be set.")

  justification: str = Field(
      description="The reason for choosing this specific device setting."
  )

  @property
  def json_metadata(self) -> serialization.SerializableData:
    """JSON-serializable metadata."""
    return self.model_dump()


class SetpointsAction(pydantic.BaseModel):
  """A flexible action model for setting any number of setpoints.

  Attributes:
    timestamp: The time the action is taken (in the building's local timezone).
    justification: The overall reason for taking this action. Includes a brief
      description of why the action is justified, as well as the desired
      outcome of the action as a whole.
    setpoints: A list of setpoints.
    validity_interval: The amount of time in minutes the setpoints should remain
      in effect before prompting for a new action.
  """

  timestamp: str = Field(description=TIMESTAMP_DESCRIPTION)

  justification: str = Field(description=JUSTIFICATION_DESCRIPTION)

  setpoints: list[DeviceSetpoint] = Field(description="A list of setpoints.")

  validity_interval: DefaultValidityIntervalOptions = Field(
      description=VALIDITY_INTERVAL_DESCRIPTION
  )

  @pydantic.field_validator("setpoints")
  @classmethod
  def validate_setpoints(
      cls, v: Sequence[DeviceSetpoint]
  ) -> Sequence[DeviceSetpoint]:
    """Ensures the setpoints are present."""
    if not v:
      raise ValueError("The setpoints list cannot be empty.")
    return v

  def find_setpoint(
      self, device_id: str, setpoint_name: str
  ) -> DeviceSetpoint | None:
    """Returns the setpoint matching the given device id and setpoint name."""
    for setpoint in self.setpoints:
      if (
          setpoint.device_id == device_id
          and setpoint.setpoint_name == setpoint_name
      ):
        return setpoint
    return None

  @property
  def json_metadata(self) -> serialization.SerializableData:
    """Serializable metadata."""
    return self.model_dump()


def create_action_model(
    custom_intervals: Sequence[int],
    model_name: str = "SetpointsActionWithCustomInterval",
) -> type[SetpointsAction]:
  """Creates an agent action model class, using custom validity intervals.

  Args:
    custom_intervals: A list of intervals in minutes. Represents the range of
      possible options the LLM has to choose from.
    model_name: The name of the action model class to be created.

  Returns:
    A Pydantic model class based on `SetpointsAction`, but defined using the
    provided set of custom validity intervals.
  """
  custom_intervals = sorted(list(set(custom_intervals)))
  ValidityIntervalOptions = Literal[*custom_intervals]  # pytype: disable=invalid-annotation  # pydantic needs it this way

  fields = {
      "validity_interval": (
          ValidityIntervalOptions,
          Field(description=VALIDITY_INTERVAL_DESCRIPTION),
      )
  }
  model = pydantic.create_model(
      model_name,
      **fields,
      __base__=SetpointsAction,
  )
  model.__doc__ = SetpointsAction.__doc__
  return model
