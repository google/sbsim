"""Action Context is an LLM output schema with awareness of the environment.

**Setpoint Content Validations**

The Action Context uses its environment to validate the content of the setpoints
in the requested action. If a given setpoint value is outside the valid range
as defined by that setpoint's action normalizer (thus exceeding the guardrails),
an error will be raised if the `clip` option is set to `False`. However, by
default, if the `clip` option is set to `True`, the setpoints will be clipped to
the bounds of the valid setpoint range, and a record of the error will be stored
(instead of being raised). For example, if the valid range for a setpoint is
[10, 20], and the LLM requests a value of 25, with clipping enabled, the value
will be clipped to 20, and a record of the error will be available in the
`guardrails_exceeded` attribute.

**Action Formatting**

The Action Context also uses its environment to convert the setpoints into a
format suitable for stepping the environment. The `ActionContext` class should
be used in conjunction with a normal continuous action `Environment`, whereas
the `HybridActionContext` class should be used with a `HybridActionEnvironment`.
Regardless of which class is used, the `get_action` method produces a properly
formatted action that can be used to step the environment.
"""

import abc
from collections.abc import Sequence, Collection
import dataclasses
import json
from typing import Any, Literal, Self

import pandas as pd
import pydantic

from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.schema import output_schema

SteppableActionType = (
    environment.NormalizedActionValues
    | hybrid_action_environment.HybridAction
)


#
# ERRORS
#


class GuardrailsExceededError(ValueError):
  """Requested setpoint value is outside the normalizer range."""


@dataclasses.dataclass(frozen=True)
class GuardrailsExceededRecord:
  """Information about a requested setpoint value that is out of range.

  Attributes:
    device_id: The device identifer.
    setpoint_name: The name of the setpoint for the given device.
    requested_value: The requested setpoint value.
    setpoint_range: The valid range of setpoint values accepted by the
      environment.
    clipped_value: The setpoint value after being clipped to the valid range.
  """
  device_id: str
  setpoint_name: str
  requested_value: float
  setpoint_range: tuple[float, float]
  clipped_value: float


#
# SCHEMA
#


class Steppable(abc.ABC):
  """An action schema that produces an action that can step an environment."""

  @abc.abstractmethod
  def get_action(
      self,
  ) -> SteppableActionType:
    """Returns an action used to step the environment."""


class ActionContext(output_schema.SetpointsAction, Steppable):
  """A `SetpointsAction` with awareness of the environment.

  This `ActionContext` class should be used in conjunction with a normal
  continuous action `Environment`.
  """

  # We are using the environment for validation of the setpoints, but it is not
  # part of the Pydantic model schema itself. Because validation runs during
  # the parent class initialization, the environment must be assigned
  # beforehand, so we use an object.__setattr__() approach. However, Pydantic
  # v2's __getattr__ intercepts access to the environment during validation,
  # which causes an AttributeError. Defining __slots__ forces the environment
  # to be managed via Python's slot mechanism, bypassing Pydantic's
  # __getattr__ and allowing it to be accessed during validation.
  #
  # TODO: b/496194630 - It might make more sense to make this a separate class,
  # instead of inheriting from the schema class.
  #
  __slots__ = ("_env", "_clip", "_guardrails_exceeded")
  _env: environment.Environment
  _clip: bool
  _guardrails_exceeded: list[GuardrailsExceededRecord]

  def __init__(
      self, env: environment.Environment, *, clip: bool = True, **kwargs
  ):
    """Initializes the instance.

    Args:
      env: The environment to use for validation.
      clip: Governs the behavior when an agent requests a setpoint value that is
        outside of the valid range. If `True`, clips the setpoint values to the
        bounds of the valid range, and logs a record of the error, but does not
        halt execution. This is the default behavior. Otherwise, if `False`,
        will raise a `GuardrailsExceededError` and halt execution.
      **kwargs: Arguments to pass to initialize the `SetpointsAction` schema.

    Raises:
      GuardrailsExceededError: If `clip` is `False` and any setpoint value is
        outside the valid range defined by the environment's normalizers.
    """
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_clip", clip)
    object.__setattr__(self, "_guardrails_exceeded", [])
    super().__init__(**kwargs)

  @classmethod
  def from_json(
      cls, txt: str, env: environment.Environment, *, clip: bool = True
  ) -> Self:
    """Creates an instance from a JSON string, while passing extra attributes.

    The LLM responds with a JSON-formatted string, but we need to pass the
    environment and clip attributes to the class constructor as well. So this
    method solves that problem.

    This method is meant to act as a replacement for Pydantic's
    `model_validate_json` method, which we would normally use, but cannot use
    with this class due to its custom `__init__` signature.

    Args:
      txt: The JSON-formatted string to parse and validate.
      env: The environment to use for validation.
      clip: Governs the behavior when an agent requests a setpoint value that is
        outside of the valid range. If `True`, clips the setpoint values to the
        bounds of the valid range, and logs a record of the error, but does not
        halt execution. This is the default behavior. Otherwise, if `False`,
        will raise a `GuardrailsExceededError` and halt execution.

    Returns:
      An instance of the class.
    """
    return cls(env=env, clip=clip, **json.loads(txt))

  @property
  def env(self) -> environment.Environment:
    """The environment."""
    return self._env

  @property
  def clip(self) -> bool:
    """Whether to clip setpoint values to the bounds of the valid range."""
    return self._clip

  @property
  def guardrails_exceeded(self) -> Collection[GuardrailsExceededRecord]:
    """A list of guardrails errors that occurred during validation."""
    return self._guardrails_exceeded

  @pydantic.model_validator(mode="after")
  def validate_setpoint_contents(self) -> Self:
    """Ensures all env action names are present, and values are in range."""
    setpoint_action_names = set()

    # CHECK SETPOINTS THAT ARE PRESENT IN THE SCHEMA
    for setpoint in self.setpoints:
      device_id = setpoint.device_id
      setpoint_name = setpoint.setpoint_name
      try:
        action_name = self.env.id_map[(device_id, setpoint_name)]
      except KeyError as err:
        raise ValueError(
            f"Setpoint for ({device_id!r}, {setpoint_name!r}) not found in the"
            f" environment"
        ) from err
      setpoint_action_names.add(action_name)

      normalizer = self.env.action_normalizers.get(setpoint_name)
      if normalizer is None:
        raise ValueError(
            f"Normalizer not found for setpoint: {action_name!r}"
        )

      setpoint_value = setpoint.setpoint_value
      setpoint_min = normalizer.setpoint_min  # min native value
      setpoint_max = normalizer.setpoint_max  # max native value
      if not (setpoint_min <= setpoint_value <= setpoint_max):
        if self._clip:
          clipped_value = max(setpoint_min, min(setpoint_value, setpoint_max))
          self._guardrails_exceeded.append(
              GuardrailsExceededRecord(
                  device_id=device_id,
                  setpoint_name=setpoint_name,
                  requested_value=setpoint_value,
                  setpoint_range=(setpoint_min, setpoint_max),
                  clipped_value=clipped_value,
              )
          )
          setpoint.setpoint_value = clipped_value
        else:
          raise GuardrailsExceededError(
              f"Value {setpoint_value} for setpoint ({device_id!r},"
              f" {setpoint_name!r}) is outside expected range [{setpoint_min},"
              f" {setpoint_max}]"
          )

    missing_action_names = set(self.env.action_names) - setpoint_action_names
    if missing_action_names:
      raise ValueError(
          "The following setpoints are expected by the environment but are"
          f" missing from the schema: {missing_action_names}"
      )

    return self

  @property
  def sorted_setpoints(self) -> Sequence[output_schema.DeviceSetpoint]:
    """The setpoints, in the same order as the environment's action names."""
    return sorted(
        self.setpoints,
        key=lambda sp: self.env.action_names.index(
            self.env.id_map[(sp.device_id, sp.setpoint_name)]
        ),
    )

  def get_action_values(self) -> environment.NormalizedActionValues:
    """Returns the normalized values used to step the `Environment`.

    Returns:
      A list of normalized action values, sorted in the same order as the
      environment's action names.
    """
    normalized_values = []
    for sp in self.sorted_setpoints:
      action_name = self.env.id_map[(sp.device_id, sp.setpoint_name)]
      normalizer = self.env.action_normalizers.get(sp.setpoint_name)
      if normalizer is None:
        raise ValueError(f"No normalizer found for setpoint: {action_name!r}.")
      normalized_values.append(normalizer.agent_value(sp.setpoint_value))
    return normalized_values

  def get_action(self) -> SteppableActionType:
    """Returns the action used to step the environment."""
    return self.get_action_values()

  @property
  def setpoint_records(self) -> list[dict[str, Any]]:
    """The setpoints as a list of records (dictionaries)."""
    return [
        {
            "timestamp": self.timestamp,
            "validity_interval": self.validity_interval,
            "justification": self.justification,
            "action_name": self.env.id_map[(sp.device_id, sp.setpoint_name)],
            "device_id": sp.device_id,
            "setpoint_name": sp.setpoint_name,
            "setpoint_value": sp.setpoint_value,
            "setpoint_justification": sp.justification,
        }
        for sp in self.sorted_setpoints
    ]

  @property
  def setpoints_df(self) -> pd.DataFrame:
    """The setpoints as a pandas DataFrame."""
    return pd.DataFrame(self.setpoint_records)

  @property
  def flattened_setpoints_record(self) -> dict[str, Any]:
    """A flattened dictionary of setpoint records. No nesting.

    The dictionary has keys for each action_name and setpoint value,
    and a second set of keys for each action_name and setpoint justification.
    """
    record = {
        "timestamp": self.timestamp,
        "validity_interval": self.validity_interval,
        "justification": self.justification,
    }
    for sp in self.sorted_setpoints:
      action_name = self.env.id_map[(sp.device_id, sp.setpoint_name)]
      record[action_name] = sp.setpoint_value
      record[f"{action_name}_justification"] = sp.justification
    return record


class HybridActionContext(ActionContext, Steppable):
  """A `SetpointsAction` with awareness of the environment.

  This class should be used in conjunction with a `HybridActionEnvironment`.
  """

  _env: hybrid_action_environment.HybridActionEnvironment

  def __init__(
      self,
      env: hybrid_action_environment.HybridActionEnvironment,
      *,
      clip: bool = True,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      env: The hybrid action environment to use for validation.
      clip: Governs the behavior when an agent requests a setpoint value that is
        outside of the valid range. If `True`, clips the setpoint values to the
        bounds of the valid range, and logs a record of the error, but does not
        halt execution. This is the default behavior. Otherwise, if `False`,
        will raise a `GuardrailsExceededError` and halt execution.
      **kwargs: Arguments to pass to initialize the `SetpointsAction` schema.

    Raises:
      GuardrailsExceededError: If `clip` is `False` and any setpoint value is
        outside the valid range defined by the environment's normalizers.
    """
    super().__init__(env=env, clip=clip, **kwargs)

  def get_hybrid_action(self) -> hybrid_action_environment.HybridAction:
    """Returns the hybrid action used to step a `HybridActionEnvironment`."""
    return self.env.convert_to_hybrid(self.get_action_values())

  def get_action(self) -> SteppableActionType:
    """Returns the action used to step the environment."""
    return self.get_hybrid_action()


def create_action_context_model(
    custom_intervals: Sequence[int],
    *,
    hybrid: bool = True,
) -> type[ActionContext]:
  """Creates an action context model class, using custom validity intervals.

  Args:
    custom_intervals: A list of intervals in minutes. Represents the range of
      possible options the LLM has to choose from.
    hybrid: Whether to create a hybrid action context model class.

  Returns:
    A Pydantic model class based on `ActionContext`, but defined using the
    provided set of custom validity intervals.
  """
  custom_intervals = sorted(set(custom_intervals))
  ValidityIntervalOptions = Literal[*custom_intervals]  # pytype: disable=invalid-annotation  # pydantic needs it this way

  fields = {
      "validity_interval": (
          ValidityIntervalOptions,
          pydantic.Field(
              description=output_schema.VALIDITY_INTERVAL_DESCRIPTION
          ),
      )
  }

  if hybrid:
    base_class = HybridActionContext
    model_name = "HybridActionContextWithCustomInterval"
  else:
    base_class = ActionContext
    model_name = "ActionContextWithCustomInterval"

  model = pydantic.create_model(
      model_name,
      **fields,
      __base__=base_class,
  )
  model.__doc__ = base_class.__doc__
  return model
