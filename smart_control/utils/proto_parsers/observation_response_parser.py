"""Parsing and Conversion Utilities for ObservationResponse protos.

Translates protos into data structures that are useful or easier to work with.
"""

from functools import cached_property  # pylint: disable=g-importing-member
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.utils import conversion_utils


ObservationResponse = smart_control_building_pb2.ObservationResponse
SingleObservationResponse = smart_control_building_pb2.SingleObservationResponse


class ObservationResponseParser:
  """Parses an ObservationResponse proto into a more usable format."""

  def __init__(self, observation_response: ObservationResponse):
    self.observation_response = observation_response

  @property
  def timestamp(self) -> pd.Timestamp:
    """Returns the current timestamp in the building's time zone."""
    return conversion_utils.proto_to_pandas_timestamp(
        self.observation_response.timestamp
    )

  def get_local_time(self, time_zone: str = 'US/Pacific') -> pd.Timestamp:
    """Returns the current local time in the building's time zone."""
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

  @property
  def outside_air_temp_measurement_name(self) -> str:
    """Returns the name of the outside air temperature measurement."""
    return 'outside_air_temperature_sensor'

  @cached_property
  def outside_air_temp(self) -> float:
    """Returns the current outside temperature.

    Assumes there is only one 'outside_air_temperature_sensor' measurement.

    Returns:
      The outside temperature in degrees Kelvin.

    Raises:
      ValueError: If zero or multiple matching measurements are found.
    """
    measurement_name = self.outside_air_temp_measurement_name

    df = self.observations_df
    if df.empty:
      raise ValueError('No observations found.')

    rows = df[df['measurement_name'] == measurement_name]
    if rows.empty:
      raise ValueError(f"No '{measurement_name}' observation found.")
    if len(rows) > 1:
      raise ValueError(f"Multiple '{measurement_name}' observations found.")

    return float(rows.iloc[0]['continuous_value'])

