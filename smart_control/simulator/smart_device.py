"""Library for creating SmartDevices.

A SmartDevice allows for observable and action fields to be defined
easily and in an extensible way. SmartDevices are meant to be used by the
simulator to produce observations and actions for an RL environment.

Any device that wants to expose observable or action fields should extend
the SmartDevice class, supplying the appropriate information to the SmartDevice
constructor.

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

import abc
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Type
import pandas as pd
from smart_control.proto import smart_control_building_pb2


class AttributeInfo(NamedTuple):
  """PODO containing information about an attribute.

  The attribute_name is the literal attribute name of an object. This name
  will be used with setattr/getattr.

  Attributes:
    attribute_name: Name of the internal attribute.
    clazz: Class of the attribute.
  """

  attribute_name: str
  clazz: Type[object]


class SmartDevice(metaclass=abc.ABCMeta):
  """Represents a SmartDevice which exposes observable/action fields."""

  def __init__(
      self,
      observable_fields: Mapping[str, AttributeInfo],
      action_fields: Mapping[str, AttributeInfo],
      device_type: smart_control_building_pb2.DeviceInfo.DeviceType,
      device_id: str,
      zone_id: Optional[str] = None,
  ):
    """Creates SmartDevice.

    Args:
      observable_fields: Fields that will be exposed as observables.
      action_fields: Fields that will be exposed as actions.
      device_type: Type of device.
      device_id: Id of device.
      zone_id: Which zone the device is in.
    """
    self._observable_fields = observable_fields
    self._action_fields = action_fields
    self._device_type = device_type
    self._device_id = device_id
    self._zone_id = zone_id
    self._action_timestamp = None
    self._observation_timestamp = None

  def device_id(self) -> str:
    """Returns device id."""
    return self._device_id

  def zone_id(self) -> Optional[str]:
    """Returns zone_id."""
    return self._zone_id

  def device_type(self) -> smart_control_building_pb2.DeviceInfo.DeviceType:
    """Returns device type."""
    return self._device_type

  def observable_field_names(self) -> Sequence[str]:
    """Returns all observable field names."""
    return self._observable_fields.keys()  # pytype: disable=bad-return-type

  def action_field_names(self) -> Sequence[str]:
    """Returns all action field names."""
    return self._action_fields.keys()  # pytype: disable=bad-return-type

  def get_observable_type(self, field_name: str) -> Type[object]:
    """Returns class type expected for field_name.

    Args:
      field_name: Name of the observable field.
    """
    return self._attribute_info(field_name, is_observable=True).clazz

  def get_action_type(self, field_name: str) -> Type[object]:
    """Returns class type expected for field_name.

    Args:
      field_name: Name of the action field.
    """
    return self._attribute_info(field_name, is_observable=False).clazz

  def _attribute_info(
      self, field_name: str, is_observable: bool
  ) -> AttributeInfo:
    """Returns mapped attribute info after checking if its valid.

    Args:
      field_name: Name of the observable field.
      is_observable: Whether field_name is for an observable. If not it is for
        an action.

    Raises:
      AttributeError: If requested field was not declared or does not exist.
    """
    if is_observable:
      field_type_name = 'observable'
      field_mapping = self._observable_fields
    else:
      field_type_name = 'action'
      field_mapping = self._action_fields

    if field_name not in field_mapping:
      raise AttributeError(
          f'Requested field: {field_name}, not set as an'
          f' {field_type_name} field.'
      )

    attribute_info = field_mapping[field_name]

    attribute_name = attribute_info.attribute_name

    if attribute_name not in dir(self):
      raise AttributeError(
          f'Requested field {field_name} maps to nonexistent attribute:'
          f' {attribute_name}.'
      )

    return attribute_info

  def get_observation(
      self, observable_field_name: str, observation_timestamp: pd.Timestamp
  ) -> Any:
    """Returns the value of an observable field.

    Args:
      observable_field_name: Name of the observable field.
      observation_timestamp: validity time of the observation.

    Raises:
      AttributeError: If requested field was not declared observable or does not
      exist.
    """
    attribute_info = self._attribute_info(
        observable_field_name, is_observable=True
    )
    attribute_name = attribute_info.attribute_name
    self._observation_timestamp = observation_timestamp

    value = getattr(self, attribute_name)

    return value

  def set_action(
      self, action_field_name: str, value: Any, action_timestamp: pd.Timestamp
  ) -> None:
    """Sets an action field with a given value.

    Args:
      action_field_name: Name of the observable field.
      value: Value to set action field to.
      action_timestamp: Timestamp of the action

    Raises:
      AttributeError: If requested field was not declared action or does
      not exist.

      ValueError: If given value is not the declared type.
    """
    attribute_info = self._attribute_info(
        action_field_name, is_observable=False
    )
    attribute_name = attribute_info.attribute_name
    attribute_type = attribute_info.clazz
    self._action_timestamp = action_timestamp

    if not isinstance(value, attribute_type):
      raise ValueError(
          f'Tried to set field: {action_field_name} with an incorrect value'
          f' type: {type(value)} is not instance of {attribute_type}.'
      )

    setattr(self, attribute_name, value)
