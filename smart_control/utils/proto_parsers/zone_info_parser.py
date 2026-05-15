"""Zone info parser makes it easier to work with ZoneInfo proto objects."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2


class ZoneInfoParser:
  """A wrapper for the ZoneInfo proto."""

  def __init__(self, zone_info: smart_control_building_pb2.ZoneInfo):
    """Initializes the instance.

    Args:
      zone_info: The ZoneInfo object to parse.
    """
    self._zone_info = zone_info

  @property
  def building_id(self) -> str:
    """The building identifier."""
    return self._zone_info.building_id

  @property
  def zone_id(self) -> str:
    """The zone identifier."""
    return self._zone_info.zone_id

  @property
  def zone_type_id(self) -> int:
    """The integer representation of the zone type enum (e.g. 1 for ROOM)."""
    return self._zone_info.zone_type

  @property
  def zone_type(self) -> str:
    """The string representation of the zone type enum (e.g. 'ROOM')."""
    return smart_control_building_pb2.ZoneInfo.ZoneType.Name(self.zone_type_id)

  @property
  def description(self) -> str:
    """The zone description."""
    return self._zone_info.zone_description

  @property
  def floor(self) -> int:
    """The floor number."""
    return self._zone_info.floor

  @property
  def area(self) -> float:
    """The area of the zone."""
    return self._zone_info.area

  @property
  def device_ids(self) -> Sequence[str]:
    """Sequence of identifiers for devices associated with the zone."""
    return self._zone_info.devices

  @property
  def as_dict(self) -> dict[str, Any]:
    """Dictionary representation of the zone info, suitable for a DataFrame."""
    return {
        'building_id': self.building_id,
        'zone_id': self.zone_id,
        'zone_type_id': self.zone_type_id,
        'zone_type': self.zone_type,
        'description': self.description,
        'floor': self.floor,
        'area': self.area,
        'device_ids': list(self.device_ids),
    }

  @property
  def devices_df(self) -> pd.DataFrame:
    """DataFrame representation of the devices associated with this zone."""
    return pd.DataFrame({
        'building_id': self.building_id,
        'zone_id': self.zone_id,
        'device_id': device_id,
    } for device_id in self.device_ids)

