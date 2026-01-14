"""Parsing and Conversion Utilities for RewardInfo protos.

Translates protos into data structures that are useful or easier to work with.
"""

import collections
from functools import cached_property  # pylint: disable=g-importing-member
from typing import Mapping, Tuple, Optional

import numpy as np
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils import conversion_utils


RewardInfo = smart_control_reward_pb2.RewardInfo

proto_to_pandas_timestamp = conversion_utils.proto_to_pandas_timestamp

_TEMP_UNIT = 'K'
_TEMP_BINS = [290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300]

_WATT_SECONDS_KWH = 1.0 / 3600.0 / 1000.0


def get_comfort_diffs(row: pd.Series) -> Tuple[Optional[str], Optional[float]]:
  """Determines whether or not the zone is in comfort range.

  Differential is calculated according to the following logic:
  If the `zone_air_temp` is between heating and cooling setpoints, then 0,
  else if `zone_air_temp` is below heating setpoint, then negative differential,
  else if `zone_air_temp` is above cooling setpoint, then positive differential.

  Args:
    row: A pandas series containing the following attribute names:
      + 'zone_air_temp'
      + 'heating_setpoint_temp'
      + 'cooling_setpoint_temp'

  Returns:
    A tuple containing the comfort label and the comfort differential.
    The comfort label is one of 'IN_RANGE', 'TOO_COLD', or 'TOO_HOT'.
    The comfort differential is the difference between the zone air temperature
    and the desired temperature range, where zero means the temp is in range,
    positive numbers are too hot, and negative numbers are too cold.
  """
  label = None
  diff = None

  if (row['zone_air_temp'] >= row['heating_setpoint_temp'] and
      row['zone_air_temp'] <= row['cooling_setpoint_temp']):
    label = 'IN_RANGE'
    diff = 0

  elif row['zone_air_temp'] < row['heating_setpoint_temp']:
    label = 'TOO_COLD'
    diff = row['zone_air_temp'] - row['heating_setpoint_temp']

  elif row['zone_air_temp'] > row['cooling_setpoint_temp']:
    label = 'TOO_HOT'
    diff = row['zone_air_temp'] - row['cooling_setpoint_temp']

  return label, diff


class RewardInfoParser:
  """Parses a RewardInfo proto into a more usable format."""

  def __init__(self, reward_info: RewardInfo):
    self.reward_info = reward_info

  # PROPERTIES AND ALIASES

  @cached_property
  def start_timestamp(self) -> pd.Timestamp:
    return proto_to_pandas_timestamp(self.reward_info.start_timestamp)

  @cached_property
  def end_timestamp(self) -> pd.Timestamp:
    return proto_to_pandas_timestamp(self.reward_info.end_timestamp)

  @cached_property
  def dt(self) -> float:
    """Returns the duration of the reward info in seconds."""
    return (self.end_timestamp - self.start_timestamp).total_seconds()

  @cached_property
  def zone_reward_infos(self) -> Mapping[str, RewardInfo.ZoneRewardInfo]:
    return self.reward_info.zone_reward_infos

  @cached_property
  def air_handler_reward_infos(self) -> Mapping[str, RewardInfo.AirHandlerRewardInfo]:  # pylint: disable=line-too-long
    return self.reward_info.air_handler_reward_infos

  @cached_property
  def boiler_reward_infos(self) -> Mapping[str, RewardInfo.BoilerRewardInfo]:
    return self.reward_info.boiler_reward_infos

  #
  # ZONE INFO
  #

  def get_zone_conditions_histogram(self, temp_unit: str = _TEMP_UNIT,
                                    temp_bins: list[float] | None = None,
                                    ):
    """Summarizes the number of zones and occupants in each temperature bin."""
    if temp_bins is None:
      temp_bins = _TEMP_BINS

    temperature_bins = np.array(temp_bins)
    temperature_count = np.zeros(len(temperature_bins))
    occupancy_count = np.zeros(len(temperature_bins))
    setpoint_count = np.zeros(len(temperature_bins))

    min_setpoint_ix = len(temperature_bins)
    max_setpoint_ix = -1

    for _, zone_reward_info in self.zone_reward_infos.items():
      zone_temp = zone_reward_info.zone_air_temperature
      bin_id = np.argmin(np.abs(temperature_bins - zone_temp))
      temperature_count[bin_id] += 1
      occupancy_count[bin_id] += zone_reward_info.average_occupancy

      heating_setpoint_temp = zone_reward_info.heating_setpoint_temperature

      bin_id = np.argmin(np.abs(temperature_bins - heating_setpoint_temp))
      if bin_id < min_setpoint_ix:
        min_setpoint_ix = bin_id

      setpoint_count[bin_id] += 1

      cooling_setpoint_temp = zone_reward_info.cooling_setpoint_temperature
      bin_id = np.argmin(np.abs(temperature_bins - cooling_setpoint_temp))
      if bin_id > max_setpoint_ix:
        max_setpoint_ix = bin_id
      setpoint_count[bin_id] += 1

    setpoint_range = ['-'] * len(temperature_bins)
    setpoint_mask = np.ones(len(temperature_bins))
    for ix in range(min_setpoint_ix, max_setpoint_ix + 1):
      setpoint_range[ix] = '+'
      setpoint_mask[ix] = 0

    occupancy_count = np.ceil(occupancy_count)

    occupants_exposed = np.multiply(occupancy_count, setpoint_mask)
    occupants_exposed = occupants_exposed.astype(int)
    temperature_count = temperature_count.astype(int)
    occupancy_count = occupancy_count.astype(int)

    return pd.DataFrame(
        {
            'count of zones': temperature_count,
            'count of occupants': occupancy_count,
            'temperature setpoint range': setpoint_range,
            'count of occupants exposed': occupants_exposed,
        },
        index=[f'{temp}°{temp_unit}' for temp in temperature_bins],
    ).T

  @cached_property
  def zone_occupancies_df(self) -> pd.DataFrame:
    """Converts a sequence of zone occupancies to a pandas dataframe.

    Returns:
      A pandas dataframe containing zone occupancy information. The dataframe is
      indexed by zone_id and sorted by zone_id. The dataframe contains the
      following columns:
        + "zone_id": The zone id as an integer.
        + "average_occupancy": The average occupancy of the zone.
        + "heating_setpoint_temp": The heating setpoint temperature of the zone.
        + "cooling_setpoint_temp": The cooling setpoint temperature of the zone.
        + "zone_air_temp": The zone air temperature.
        + "comfort_label": The comfort label of the zone.
        + "comfort_diff": The comfort differential of the zone.
    """
    records = []
    for zone_id, info in self.zone_reward_infos.items():
      records.append({
          'zone_id': zone_id,
          'average_occupancy': info.average_occupancy,
          'heating_setpoint_temp': info.heating_setpoint_temperature,
          'cooling_setpoint_temp': info.cooling_setpoint_temperature,
          'zone_air_temp': info.zone_air_temperature,
      })

    df = pd.DataFrame(records)
    df.set_index('zone_id', inplace=True)
    df.sort_index(inplace=True)
    df['zone_air_temp'] = df['zone_air_temp'].round(1)
    df[['comfort_label', 'comfort_diff']] = df.apply(get_comfort_diffs, axis=1,
                                                     result_type='expand')

    # make the label categorical, so a pivot table made from this dataframe will
    # retain a row for each of the label values, even if they are not present:
    # the order corresponds to the row sort order in the pivot table...
    categories_in_sort_order = ['TOO_HOT', 'IN_RANGE', 'TOO_COLD']
    df['comfort_label'] = pd.Categorical(
        df['comfort_label'], categories=categories_in_sort_order
    )

    return df

  @cached_property
  def num_zones(self) -> int:
    return len(self.zone_occupancies_df)

  @cached_property
  def total_occupancy(self) -> int:
    return int(self.zone_occupancies_df['average_occupancy'].sum())

  @cached_property
  def num_occupants_comfortable(self) -> int:
    comfortable_zones_df = self.zone_occupancies_df[
        self.zone_occupancies_df['comfort_label'] == 'IN_RANGE'
    ]
    return int(comfortable_zones_df['average_occupancy'].sum())

  @cached_property
  def num_occupants_uncomfortable(self) -> int:
    uncomfortable_zones_df = self.zone_occupancies_df[
        self.zone_occupancies_df['comfort_label'] != 'IN_RANGE'
    ]
    return int(uncomfortable_zones_df['average_occupancy'].sum())

  @cached_property
  def occupant_comfort_histogram(self) -> dict[str, int]:
    """The number of occupants below, in, and above comfort setpoint range.

    Returns:
      A dictionary mapping of comfort range labels like:
        `{'TOO_HOT': 0, 'IN_RANGE': 10, 'TOO_COLD': 0}`
    """
    groupby = self.zone_occupancies_df.groupby('comfort_label', observed=False)
    return groupby['average_occupancy'].sum().to_dict()

  #
  # ENERGY CONSUMPTION
  #

  def get_energy_consumption(self) -> Mapping[str, float]:
    """Energy consumption in kWh for ac, blower, pump, and nat gas heating."""

    energy_use = collections.defaultdict(float)

    for air_handler_id in self.air_handler_reward_infos:
      energy_use['air_handler_blower_electricity'] += (
          self.air_handler_reward_infos[
              air_handler_id
          ].blower_electrical_energy_rate
          * self.dt
          * _WATT_SECONDS_KWH
      )
      energy_use['air_handler_air_conditioning'] += (
          self.air_handler_reward_infos[
              air_handler_id
          ].air_conditioning_electrical_energy_rate
          * self.dt
          * _WATT_SECONDS_KWH
      )

    for boiler_id in self.boiler_reward_infos:
      energy_use['boiler_natural_gas_heating_energy'] += (
          self.boiler_reward_infos[
              boiler_id
          ].natural_gas_heating_energy_rate
          * self.dt
          * _WATT_SECONDS_KWH
      )
      energy_use['boiler_pump_electrical_energy'] += (
          self.boiler_reward_infos[boiler_id].pump_electrical_energy_rate
          * self.dt
          * _WATT_SECONDS_KWH
      )

    return energy_use

  @cached_property
  def energy_consumption_df(self) -> pd.DataFrame:
    """Compiles a dataframe of energy consumption for each device.

    Descriptions come from the proto definitions.

    Returns:
      A pandas dataframe containing energy consumption information. Contains
      the following columns:
        + 'device_type: The device type (AC or HWS).
        + 'device_id: The device id.
        + 'metric': The energy consumption metric name.
        + 'description': A description of the energy consumption metric.
        + 'value': The energy consumption (rate of consumption per second).
        + 'unit': The energy consumption unit.
    """
    records = []

    # AIR HANDLER REWARDS:
    for device_id, ac_reward_info in self.air_handler_reward_infos.items():
      device_type = 'AHU'

      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'blower_electrical_energy_rate',
          'description': 'Cumulative electrical power in W applied to blowers.',
          'value': ac_reward_info.blower_electrical_energy_rate,
          'unit': 'W'
      })
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'air_conditioning_electrical_energy_rate',
          'description': (
              'Cumulative electrical energy rate applied in W for air '
              'conditioning. This represents the total power applied for '
              'running refrigeration or heat pump cycles (includes running a '
              'compressor and pumps to recirculate refrigerant).'
          ),
          'value': ac_reward_info.air_conditioning_electrical_energy_rate,
          'unit': 'W'
      })

    # HWS REWARDS:
    for device_id, hws_reward_info in self.boiler_reward_infos.items():
      device_type = 'HWS'

      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'pump_electrical_energy_rate',
          'description': (
              'Cumulative electrical power in W for water recirculation pumps.'
          ),
          'value': hws_reward_info.pump_electrical_energy_rate,
          'unit': 'W'
      })

      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'natural_gas_heating_energy_rate',
          'description': (
              'Energy rate consumed in W by natural gas for heating water.'
          ),
          'value': hws_reward_info.natural_gas_heating_energy_rate,
          'unit': 'W',
      })

    df = pd.DataFrame(records)
    if df.empty:
      raise ValueError('No energy consumption data found.')
    return df

  @cached_property
  def energy_consumption_df_watts(self) -> pd.DataFrame:
    """A version of the energy consumption data, where the unit is Watts."""
    df = self.energy_consumption_df.copy()
    # filter out non-watts rows (in case we see some in the future)
    df = df[df['unit'] == 'W']
    # get opinionated about the units, which are all currently in watts
    df = df.rename(columns={'value': 'rate_watts'})
    df = df.drop(columns=['unit'], errors='ignore')
    # calculate the energy consumption in kWh:
    df['consumption_kwh'] = df['rate_watts'] * self.dt * _WATT_SECONDS_KWH
    return df
