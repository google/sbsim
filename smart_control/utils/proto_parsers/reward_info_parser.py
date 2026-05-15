"""Parsing and Conversion Utilities for RewardInfo protos.

Translates protos into data structures that are useful or easier to work with.
"""

import collections
from collections.abc import Mapping, Sequence
from functools import cached_property  # pylint: disable=g-importing-member
from typing import Any

import numpy as np
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import conversion_utils
from smart_buildings.smart_control.utils import temperature_conversion as tc

proto_to_pandas_timestamp = conversion_utils.proto_to_pandas_timestamp

WATT_SECONDS_KWH = conversion_utils._WATT_SECONDS_KWH  # pylint: disable=protected-access

TEMP_UNIT = tc.TempUnit.KELVIN
TEMP_BINS: Sequence[float] = (
    290.0,
    291.0,
    292.0,
    293.0,
    294.0,
    295.0,
    296.0,
    297.0,
    298.0,
    299.0,
    300.0,
)


# Suffix appended to measurement names to indicate the floor number.
FLOOR_PREFIX = '@floor'

# Prefix used in DataFrame columns to identify occupancy metrics for a
# specific floor.
OCCUPANCY_AT_FLOOR_PREFIX = 'occ@floor'

# The exact string identifier used for zone air temperature sensors.
ZONE_AIR_TEMPERATURE_SENSOR = 'zone_air_temperature_sensor'


def get_comfort_diffs(
    row: pd.Series,
    use_magnitude_labels: bool = False,
    label_max_degrees: int | None = 5,
) -> tuple[float, str]:
  """Calculates a comfort label and differential for each zone.

  Args:
    row: A `pandas.Series` containing the following attribute / column names:
      + 'zone_air_temp'
      + 'heating_setpoint_temp'
      + 'cooling_setpoint_temp'
    use_magnitude_labels: If True, the label will include the magnitude of the
      temperature differential.
    label_max_degrees: If provided and use_magnitude_labels is True, specifies
      the maximum number of degrees outside comfort range to be used when
      compiling the label. Must be a positive int.

  Returns:
    A tuple containing the comfort differential and corresponding label.

    The comfort differential is the difference between the zone air temperature
    and the desired temperature range, where zero means the temp is in range,
    positive numbers are too hot, and negative numbers are too cold.

    The comfort label is one of: 'IN_RANGE', 'TOO_COLD', or 'TOO_HOT'. If
    use_magnitude_labels is True, the label is appended with '_X', where X is
    the number of degrees outside of comfort range, represented as a rounded
    absolute integer value (potentially capped by label_max_degrees).
  """
  zone_air_temp = float(row['zone_air_temp'])
  comfort_min = float(row['heating_setpoint_temp'])
  comfort_max = float(row['cooling_setpoint_temp'])
  if comfort_min >= comfort_max:
    raise ValueError('Invalid setpoint range. Expecting heating < cooling.')

  if comfort_min <= zone_air_temp <= comfort_max:
    label = 'IN_RANGE'
    diff = 0.0
  elif zone_air_temp < comfort_min:
    label = 'TOO_COLD'
    diff = zone_air_temp - comfort_min
  elif zone_air_temp > comfort_max:
    label = 'TOO_HOT'
    diff = zone_air_temp - comfort_max
  else:
    raise ValueError('Invalid temperature values.')

  if use_magnitude_labels and label != 'IN_RANGE':
    degrees_outside_range = round(abs(diff))
    if label_max_degrees is not None:
      degrees_outside_range = min(degrees_outside_range, label_max_degrees)
    label = f'{label}_{degrees_outside_range}'

  return diff, label


class RewardInfoParser:
  """A parser for RewardInfo protos, converting them into more usable data structures."""

  def __init__(
      self,
      reward_info: reward_pb2.RewardInfo,
      comfort_diff_params: Mapping[str, Any] | None = None,
  ):
    """Initializes the RewardInfoParser.

    Args:
      reward_info: The RewardInfo proto to parse.
      comfort_diff_params: A dictionary of parameters to pass to the
        get_comfort_diffs function (default is None).
    """
    self.reward_info = reward_info
    self.comfort_diff_params = comfort_diff_params or {}

  def _setup_temp_params(
      self,
      temp_unit: tc.TempUnit | str | None = None,
      temp_bins: Sequence[float] | None = None,
  ) -> tuple[tc.TempUnit, Sequence[float], tc.TempConversionFunction | None]:
    """Validates and sets up temperature units and bins."""
    if temp_unit is not None:
      unit = tc.assign_temp_unit(temp_unit)
    else:
      unit = TEMP_UNIT

    temp_convert = tc.assign_kelvin_conversion_function(unit)

    if temp_bins is None:
      bins = TEMP_BINS
      if temp_convert is not None:
        bins = [temp_convert(b) for b in bins]
    else:
      bins = temp_bins

    return unit, bins, temp_convert

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
  def zone_reward_infos(
      self,
  ) -> Mapping[str, reward_pb2.RewardInfo.ZoneRewardInfo]:
    return self.reward_info.zone_reward_infos

  @cached_property
  def air_handler_reward_infos(
      self,
  ) -> Mapping[str, reward_pb2.RewardInfo.AirHandlerRewardInfo]:
    return self.reward_info.air_handler_reward_infos

  @cached_property
  def boiler_reward_infos(
      self,
  ) -> Mapping[str, reward_pb2.RewardInfo.BoilerRewardInfo]:
    return self.reward_info.boiler_reward_infos

  @cached_property
  def heat_pump_reward_infos(
      self,
  ) -> Mapping[str, reward_pb2.RewardInfo.HeatPumpRewardInfo]:
    return self.reward_info.heat_pump_reward_infos

  #
  # ZONE INFO
  #

  def get_zone_conditions_histogram_by_floor(
      self,
      zones: Sequence[smart_control_building_pb2.ZoneInfo],
      temp_unit: tc.TempUnit | str | None = None,
      temp_bins: Sequence[float] | None = None,
  ) -> pd.DataFrame:
    """Generates a histogram DataFrame of building zone conditions by temp bin.

    This function aggregates telemetry data from multiple building zones. It
    bins the current air temperature of each zone by floor, calculates the total
    occupancy for each temperature bin, and determines how many occupants are
    exposed to temperatures outside the established heating/cooling setpoints.

    Args:
        zones: A list of Protobuf ZoneInfo objects containing metadata (like the
          floor number) for each zone in the building.
        temp_unit: The unit of temperature to use (default is Kelvin).
        temp_bins: The temperature bins to use for the histogram (default is
          TEMP_BINS).

    Returns:
        pd.DataFrame: A DataFrame indexed by the `temperature_bins`.
            Columns include:
            - 'occupancy_count': Total occupants currently experiencing this
            temperature.
            - 'setpoint_mask': 0 if the temperature is within the global
            setpoint
            range,
              -1 if below the heating setpoint, 1 if above the cooling setpoint.
            - 'setpoint_range': String visualization ('-' for out of bounds, '+'
            for in bounds).
            - 'exposed_count': Number of occupants exposed to out-of-bounds
            temperatures
              (positive for too hot, negative for too cold).
            - 'floor_X' (multiple): Normalized distribution of zone temperatures
            for floor X.
    """
    temp_unit, temp_bins, temp_convert = self._setup_temp_params(
        temp_unit, temp_bins
    )

    # Convert bins to a numpy array for vectorized distance calculations later.
    bins = np.array(temp_bins)
    num_bins = len(bins)

    # Create a fast lookup dictionary to map a zone's ID to its floor number.
    zone_floor_map = {zone.zone_id: zone.floor for zone in zones}

    # Use a defaultdict to dynamically allocate arrays for floors as we
    # encounter them.
    # This safely handles missing floors, negative floors (basements), or sparse
    # floor maps.
    temperature_count_by_floor = collections.defaultdict(
        lambda: np.zeros(num_bins, dtype=float)
    )

    # Array to accumulate total occupancy per temperature bin across the whole
    # building.
    occupancy_count = np.zeros(num_bins)

    # Variables to track the absolute lowest heating setpoint and highest
    # cooling setpoint across all zones, represented as indices of the `bins`
    # array.
    min_setpoint_ix = None
    max_setpoint_ix = None

    def get_bin_idx(val: float) -> int:
      """Helper function to find the index of the temperature bin closest to `val`."""
      return int(np.argmin(np.abs(bins - val)))

    # --- Step 1: Accumulate Zone Data ---
    for zone_id, zone_reward in self.reward_info.zone_reward_infos.items():
      # Retrieve the floor for this zone. Default to 0 if the zone metadata is
      # missing.
      floor = zone_floor_map.get(zone_id, 0)

      # Find which temperature bin this zone's current air temperature falls
      # into, and increment the count for this specific floor.
      zone_air_temp = zone_reward.zone_air_temperature
      heating_setpoint_temp = zone_reward.heating_setpoint_temperature
      cooling_setpoint_temp = zone_reward.cooling_setpoint_temperature
      if temp_convert:
        zone_air_temp = temp_convert(zone_air_temp)
        heating_setpoint_temp = temp_convert(heating_setpoint_temp)
        cooling_setpoint_temp = temp_convert(cooling_setpoint_temp)

      temp_idx = get_bin_idx(zone_air_temp)
      temperature_count_by_floor[floor][temp_idx] += 1

      # Add this zone's occupants to the total count for this temperature bin.
      occupancy_count[temp_idx] += zone_reward.average_occupancy

      # Find which bins correspond to this zone's specific setpoints.
      heat_idx = get_bin_idx(heating_setpoint_temp)
      cool_idx = get_bin_idx(cooling_setpoint_temp)

      # Expand the global acceptable setpoint bounds if this zone's bounds are
      # wider.
      if min_setpoint_ix is None:
        min_setpoint_ix = heat_idx
        max_setpoint_ix = cool_idx
      else:
        min_setpoint_ix = min(min_setpoint_ix, heat_idx)
        max_setpoint_ix = max(max_setpoint_ix, cool_idx)

    # --- Step 2: Vectorized Setpoint Masking ---
    # Initialize mask arrays. By default, assume all temperatures are
    # out-of-bounds ("-").
    setpoint_mask = np.zeros(num_bins, dtype=int)
    setpoint_range = np.full(num_bins, '-', dtype=object)

    # If we actually processed zones (meaning max_setpoint_ix is not None),
    # slice the arrays to reflect the calculated global setpoint bounds.
    if max_setpoint_ix is not None:
      setpoint_mask[:min_setpoint_ix] = (
          -1
      )  # Temps below global heating setpoint
      setpoint_mask[max_setpoint_ix + 1 :] = (
          1  # Temps above global cooling setpoint
      )
      setpoint_range[min_setpoint_ix : max_setpoint_ix + 1] = (
          '+'  # Acceptable comfort range
      )

    # --- Step 3: Calculate Exposed Occupancy ---
    # Round up occupancy (you can't have a fraction of a person) and cast to
    # integer.
    occupancy_count = np.ceil(occupancy_count).astype(int)

    # Multiply occupancy by the setpoint mask.
    # Result: 0 = comfortable, negative values = cold occupants,
    # positive values = hot occupants.
    occupants_exposed = (occupancy_count * setpoint_mask).astype(int)

    # --- Step 4: Assemble the Base DataFrame ---
    table_rows = {
        'occupancy_count': occupancy_count,
        'setpoint_mask': setpoint_mask,
        'setpoint_range': setpoint_range,
        'exposed_count': occupants_exposed,
    }

    # --- Step 5: Normalize and Append Floor Distributions ---
    # Sort the dictionary by floor number to ensure predictable column ordering.
    for floor, count_arr in sorted(temperature_count_by_floor.items()):
      total_floor_count = np.sum(count_arr)
      # Normalize the array so it represents a probability
      # distribution (summing to 1.0)
      if total_floor_count > 0:
        count_arr = count_arr / total_floor_count

      # Add this floor's normalized distribution to the final table.
      table_rows[f'{OCCUPANCY_AT_FLOOR_PREFIX}{floor}'] = count_arr

    # Return the fully constructed DataFrame, using the specific
    # temperature bins as the row index.
    return pd.DataFrame(table_rows, index=bins)

  def get_zone_conditions_histogram(
      self,
      temp_unit: tc.TempUnit | str | None = None,
      temp_bins: Sequence[float] | None = None,
  ) -> pd.DataFrame:
    """Summarizes the number of zones and occupants in each temperature bin.

    Zone temperatures are assigned to the bin with the closest numerical value.

    Args:
      temp_unit: The unit of temperature to use (default is Kelvin).
      temp_bins: The temperature bins to use for the histogram (default is
        TEMP_BINS).

    Returns:
      A pandas dataframe containing the number of zones and occupants in each
      temperature bin. The dataframe is indexed by temperature and contains the
      following columns:
        + 'count of zones': The number of zones in each temperature bin.
        + 'count of occupants': The number of occupants in each temperature bin.
        + 'temperature setpoint range': A string indicating the temperature
          setpoint range ('+' indicates in range, '-' indicates out of range).
        + 'count of occupants exposed': The number of occupants exposed to
          uncomfortable temperatures.
      The dataframe is transposed so that the index is the metrics and the
      columns are the temperature bins.
    """
    temp_unit, temp_bins, temp_convert = self._setup_temp_params(
        temp_unit, temp_bins
    )

    temperature_bins = np.array(temp_bins)
    temperature_count = np.zeros(len(temperature_bins))
    occupancy_count = np.zeros(len(temperature_bins))
    setpoint_count = np.zeros(len(temperature_bins))

    min_setpoint_ix = len(temperature_bins)
    max_setpoint_ix = -1

    for _, zone_reward_info in self.zone_reward_infos.items():
      zone_temp = zone_reward_info.zone_air_temperature
      heating_setpoint_temp = zone_reward_info.heating_setpoint_temperature
      cooling_setpoint_temp = zone_reward_info.cooling_setpoint_temperature
      if temp_convert is not None:
        zone_temp = temp_convert(zone_temp)
        heating_setpoint_temp = temp_convert(heating_setpoint_temp)
        cooling_setpoint_temp = temp_convert(cooling_setpoint_temp)

      bin_id = np.argmin(np.abs(temperature_bins - zone_temp))
      temperature_count[bin_id] += 1
      occupancy_count[bin_id] += zone_reward_info.average_occupancy

      bin_id = np.argmin(np.abs(temperature_bins - heating_setpoint_temp))
      if bin_id < min_setpoint_ix:
        min_setpoint_ix = bin_id

      setpoint_count[bin_id] += 1

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
        index=[
            f'{temp}{temp_unit.deg_symbol}{temp_unit.abbrev}'
            for temp in temperature_bins
        ],
    ).T

  @cached_property
  def zone_conditions_histogram(self) -> pd.DataFrame:
    return self.get_zone_conditions_histogram()

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
    df[['comfort_diff', 'comfort_label']] = df.apply(
        get_comfort_diffs, axis=1, result_type='expand',
        **self.comfort_diff_params
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

  def watts_to_kwh(self, watts: float) -> float:
    """Converts watts to kWh for the given device."""
    return watts * self.dt * WATT_SECONDS_KWH

  def get_energy_consumption(self) -> Mapping[str, float]:
    """Returns a dictionary of energy consumption, in kWh, for each source."""

    energy_use = collections.defaultdict(float)

    # AIR HANDLER REWARDS:
    for ahu_info in self.air_handler_reward_infos.values():
      energy_use['air_handler_blower_electrical_energy'] += self.watts_to_kwh(
          ahu_info.blower_electrical_energy_rate
      )
      energy_use['air_handler_air_conditioning_electrical_energy'] += self.watts_to_kwh(  # pylint: disable=line-too-long
          ahu_info.air_conditioning_electrical_energy_rate
      )

    # BOILER REWARDS:
    for blr_info in self.boiler_reward_infos.values():
      energy_use['boiler_natural_gas_heating_energy'] += self.watts_to_kwh(
          blr_info.natural_gas_heating_energy_rate
      )
      energy_use['boiler_pump_electrical_energy'] += self.watts_to_kwh(
          blr_info.pump_electrical_energy_rate
      )

    # HEAT PUMP REWARDS:
    for ashp_info in self.heat_pump_reward_infos.values():
      energy_use['heat_pump_electricity_heating_energy'] += self.watts_to_kwh(
          ashp_info.electricity_heating_energy_rate
      )
      energy_use['heat_pump_pump_electrical_energy'] += self.watts_to_kwh(
          ashp_info.pump_electrical_energy_rate
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
    for device_id, ahu_reward_info in self.air_handler_reward_infos.items():
      device_type = 'AHU'
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'blower_electrical_energy_rate',
          'description': 'Cumulative electrical power in W applied to blowers.',
          'value': ahu_reward_info.blower_electrical_energy_rate,
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
          'value': ahu_reward_info.air_conditioning_electrical_energy_rate,
          'unit': 'W'
      })

    # BOILER REWARDS:
    for device_id, blr_reward_info in self.boiler_reward_infos.items():
      device_type = 'BLR'
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'pump_electrical_energy_rate',
          'description': (
              'Cumulative electrical power in W for water recirculation pumps.'
          ),
          'value': blr_reward_info.pump_electrical_energy_rate,
          'unit': 'W'
      })
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'natural_gas_heating_energy_rate',
          'description': (
              'Energy rate consumed in W by natural gas for heating water.'
          ),
          'value': blr_reward_info.natural_gas_heating_energy_rate,
          'unit': 'W',
      })

    # HEAT PUMP REWARDS:
    for device_id, ashp_reward_info in self.heat_pump_reward_infos.items():
      device_type = 'ASHP'
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'electricity_heating_energy_rate',
          'description': (
              'Energy rate consumed in W by electricity for heating water.'
          ),
          'value': ashp_reward_info.electricity_heating_energy_rate,
          'unit': 'W',
      })
      records.append({
          'device_type': device_type,
          'device_id': device_id,
          'metric': 'pump_electrical_energy_rate',
          'description': (
              'Cumulative electrical power in W for water recirculation pumps.'
          ),
          'value': ashp_reward_info.pump_electrical_energy_rate,
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
    df['consumption_kwh'] = self.watts_to_kwh(df['rate_watts'])
    return df
