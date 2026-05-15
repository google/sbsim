"""Device info parser makes it easier to work with DeviceInfo proto objects."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any

import pandas as pd

from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2


@dataclasses.dataclass(frozen=True)
class BaseDeviceField:
  """Schema for a single device field record.

  Attributes:
    device_id: The unique identifier of the device.
    field_name: The name of the field.
    field_type_id: The integer ID of the field's value type
      (from building_pb2.DeviceInfo.ValueType).
    field_type: String representation of the field type enum.
    as_dict: Dictionary representation of the device field record.
  """
  device_id: str
  field_name: str
  field_type_id: int

  @property
  def field_type(self) -> str:
    """String representation of the field type enum."""
    return building_pb2.DeviceInfo.ValueType.Name(self.field_type_id)

  @property
  def as_dict(self) -> dict[str, Any]:
    """Dictionary representation of the device field record."""
    return {
        'device_id': self.device_id,
        'field_name': self.field_name,
        'field_type_id': self.field_type_id,
        'field_type': self.field_type,
    }


@dataclasses.dataclass(frozen=True)
class ObservableField(BaseDeviceField):
  """Schema for a single observable field record."""


@dataclasses.dataclass(frozen=True)
class ActionField(BaseDeviceField):
  """Schema for a single action field record."""


@dataclasses.dataclass(frozen=True)
class DeviceField(BaseDeviceField):
  """Schema for a single device field record (observable and/or actionable).

  Attributes:
    is_actionable: Whether the field is actionable.
    is_observable: Whether the field is observable.
  """
  is_observable: bool
  is_actionable: bool

  @property
  def as_dict(self) -> dict[str, Any]:
    """Dictionary representation of the device field record."""
    return {
        'device_id': self.device_id,
        'field_name': self.field_name,
        'field_type_id': self.field_type_id,
        'field_type': self.field_type,
        'is_observable': self.is_observable,
        'is_actionable': self.is_actionable,
    }


class DeviceInfoParser:
  """A wrapper for the DeviceInfo proto."""

  def __init__(self, device_info: building_pb2.DeviceInfo):
    """Initializes the instance.

    Args:
      device_info: The DeviceInfo object to parse.
    """
    self._device_info = device_info

  @property
  def device_id(self) -> str:
    """The unique device identifier."""
    return self._device_info.device_id

  @property
  def device_type_id(self) -> int:
    """Integer representation of the device type enum."""
    return self._device_info.device_type

  @property
  def device_type(self) -> str:
    """String representation of the device type enum."""
    return building_pb2.DeviceInfo.DeviceType.Name(self.device_type_id)

  @property
  def zone_id(self) -> str:
    """The zone identifier where this device is located.

    In practice, this is often an empty string, because the zone to device
    mappings are typically located in the zone info's devices / device_ids
    property.
    """
    return self._device_info.zone_id

  @property
  def code(self) -> str:
    """A human-readable identifier for the device."""
    return self._device_info.code

  @property
  def namespace(self) -> str:
    """The device namespace."""
    return self._device_info.namespace

  @property
  def as_dict(self) -> dict[str, Any]:
    """Dictionary representation of the device info, suitable for a DataFrame."""
    return {
        'device_id': self.device_id,
        'device_type_id': self.device_type_id,
        'device_type': self.device_type,
        'namespace': self.namespace,
        'code': self.code,
        'zone_id': self.zone_id,
    }

  # DEVICE FIELDS

  @functools.cached_property
  def observable_fields(self) -> Sequence[ObservableField]:
    """Fields in the observation space."""
    return tuple(sorted(
        (ObservableField(
            device_id=self.device_id,
            field_name=k,
            field_type_id=v,
        ) for k, v in self._device_info.observable_fields.items()),
        key=lambda x: x.field_name
    ))

  @functools.cached_property
  def action_fields(self) -> Sequence[ActionField]:
    """Fields in the action space."""
    return tuple(sorted(
        (ActionField(
            device_id=self.device_id,
            field_name=k,
            field_type_id=v,
        ) for k, v in self._device_info.action_fields.items()),
        key=lambda x: x.field_name
    ))

  @functools.cached_property
  def fields(self) -> Sequence[DeviceField]:
    """All device fields (observable and/or actionable)."""
    all_field_names = sorted(
        set(self._device_info.observable_fields.keys())
        | set(self._device_info.action_fields.keys())
    )
    records = []
    for field_name in all_field_names:
      field_type_id = (
          self._device_info.observable_fields[field_name]
          if field_name in self._device_info.observable_fields
          else self._device_info.action_fields[field_name]
      )
      records.append(
          DeviceField(
              device_id=self.device_id,
              field_name=field_name,
              field_type_id=field_type_id,
              is_observable=field_name in self._device_info.observable_fields,
              is_actionable=field_name in self._device_info.action_fields,
          )
      )
    return tuple(records)

  @property
  def observable_fields_df(self) -> pd.DataFrame:
    """The observable fields, in DataFrame format."""
    return pd.DataFrame(r.as_dict for r in self.observable_fields)

  @property
  def action_fields_df(self) -> pd.DataFrame:
    """The action fields, in DataFrame format."""
    return pd.DataFrame(r.as_dict for r in self.action_fields)

  @property
  def fields_df(self) -> pd.DataFrame:
    """All device fields (observable and/or actionable), in DataFrame format."""
    return pd.DataFrame(r.as_dict for r in self.fields)

