"""Parsing and Conversion Utilities for ObservationResponse protos.

Translates protos into data structures that are useful or easier to work with.
"""

from functools import cached_property  # pylint: disable=g-importing-member
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.utils import conversion_utils


ObservationResponse = smart_control_building_pb2.ObservationResponse
SingleObservationResponse = smart_control_building_pb2.SingleObservationResponse

OUTSIDE_AIR_TEMP = 'outside_air_temperature_sensor'


class ObservationResponseParser:
  """Parses an ObservationResponse proto into a more usable format."""

  def __init__(
      self,
      observation_response: ObservationResponse,
      outside_air_temp_measurement_name: str = OUTSIDE_AIR_TEMP,
  ):
    self.observation_response = observation_response
    self.outside_air_temp_measurement_name = outside_air_temp_measurement_name

  @property
  def timestamp(self) -> pd.Timestamp:
    """Returns the current local time, assumed to be in UTC (see proto)."""
    return conversion_utils.proto_to_pandas_timestamp(
        self.observation_response.timestamp
    )

  def get_local_time(self, time_zone: str = 'US/Pacific') -> pd.Timestamp:
    """Returns the current local time in the building's time zone."""
    if self.timestamp.tz is None or str(self.timestamp.tz) != 'UTC':
      raise ValueError('Timestamp expected to be in UTC.')

    return self.timestamp.tz_convert(time_zone)

  @cached_property
  def single_observation_responses(self) -> list[SingleObservationResponse]:
    return list(self.observation_response.single_observation_responses)

  @cached_property
  def observations_df(self) -> pd.DataFrame:
    """Converts an sequence of observations to a pandas dataframe."""
    records = []
    for response in self.single_observation_responses:
      request = response.single_observation_request
      records.append({
          'device_id': request.device_id,
          'measurement_name': request.measurement_name,
          'is_valid': response.observation_valid,
          'continuous_value': response.continuous_value,
      })
    return pd.DataFrame(records)

  @cached_property
  def outside_air_temps_df(self) -> pd.DataFrame:
    """Returns a DataFrame of current outside temperature(s).

    Assumes there can be more than one 'outside_air_temperature_sensor'
    measurement, for example, one for each air handler unit.

    Returns:
      The outside air temperature sensors in DataFrame format.

    Raises:
      ValueError: If zero matching measurements are found.
    """
    measurement_name = self.outside_air_temp_measurement_name

    df = self.observations_df
    if df.empty:
      raise ValueError('No observations found.')

    rows = df[df['measurement_name'].str.contains(measurement_name)]
    if rows.empty:
      raise ValueError(f"No '{measurement_name}' observation found.")

    return rows

  @cached_property
  def outside_air_temp(self) -> float:
    """Returns the (average) outside temperature in degrees Kelvin."""
    return float(self.outside_air_temps_df['continuous_value'].mean())


