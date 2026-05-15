"""Factories and helper functions for proto parser tests."""

import pandas as pd
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils import test_utils


TIME_ZONE = 'US/Pacific'
START_TIMESTAMP = pd.Timestamp('2022-03-13 00:00:00', tz=TIME_ZONE)
END_TIMESTAMP = pd.Timestamp('2022-03-13 00:05:00', tz=TIME_ZONE)


def get_reward_info(
    start_timestamp: pd.Timestamp = START_TIMESTAMP,
    end_timestamp: pd.Timestamp = END_TIMESTAMP,
) -> smart_control_reward_pb2.RewardInfo:
  """Returns a RewardInfo object, for testing purposes."""

  # FYI the usual comfort range is between 293 and 297 K
  zone_temp_occupancies = [
      # zone_id, zone_air_temp, zone_occupancy
      ('zone_0', 295.0, 8.0),  # IN_RANGE  (71.33°F)
      ('zone_1', 292.0, 4.0),  # TOO_COLD_1  (65.93°F)
      ('zone_2', 299.0, 2.0),  # TOO_HOT_2  (78.53°F)
  ]

  air_handler_energies = [
      # ahu_id, blower_electrical_energy_rate, ac_electrical_energy_rate
      ('air_handler_0', 23.0, 15.0),
      ('air_handler_1', 26.0, 22.0),
  ]

  boiler_energies = [
      # hws_id, natural_gas_heating_energy_rate, pump_electrical_energy_rate
      ('boiler_0', 200.1, 2.3),
  ]

  heat_pump_energies = [
      # ashp_id, electricity_heating_energy_rate, pump_electrical_energy_rate
      ('heat_pump_0', 150.0, 20.0),
  ]

  return test_utils.get_test_reward_info(
      zone_temp_occupancies=zone_temp_occupancies,
      air_handler_energies=air_handler_energies,
      boiler_energies=boiler_energies,
      heat_pump_energies=heat_pump_energies,
      start_timestamp=start_timestamp,
      end_timestamp=end_timestamp,
  )
