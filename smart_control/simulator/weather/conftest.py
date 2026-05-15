"""Test fixtures and factories for weather related functionality."""

import pandas as pd
from smart_buildings.smart_control.simulator import weather_controller

ReplayWeatherController = weather_controller.ReplayWeatherController


# FACTORIES


def create_replay_weather_controller(
    csv_filepath: str | None = None,
) -> ReplayWeatherController:
  """Returns a default ReplayWeatherController object for test purposes."""
  return ReplayWeatherController(
      local_weather_path=csv_filepath,
      convection_coefficient=100.0,
      humidity_column='Humidity',
  )


# EXAMPLE DATA


START_TIMESTAMP = pd.Timestamp('2023-08-29 08:29:00', tz='UTC')
EXPECTED_FORECAST_PERIODS = [
    {
        'start_timestamp': pd.Timestamp('2023-08-29 08:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 09:00:00+0000', tz='UTC'),
        'temp': 289.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 09:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 10:00:00+0000', tz='UTC'),
        'temp': 289.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 10:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 11:00:00+0000', tz='UTC'),
        'temp': 288.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 11:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 12:00:00+0000', tz='UTC'),
        'temp': 288.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 12:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 13:00:00+0000', tz='UTC'),
        'temp': 288.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 13:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 14:00:00+0000', tz='UTC'),
        'temp': 288.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 14:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 15:00:00+0000', tz='UTC'),
        'temp': 287.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 15:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 16:00:00+0000', tz='UTC'),
        'temp': 289.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 16:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 17:00:00+0000', tz='UTC'),
        'temp': 291.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 17:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 18:00:00+0000', tz='UTC'),
        'temp': 293.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 18:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 19:00:00+0000', tz='UTC'),
        'temp': 294.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 19:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 20:00:00+0000', tz='UTC'),
        'temp': 296.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 20:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 21:00:00+0000', tz='UTC'),
        'temp': 297.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 21:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 22:00:00+0000', tz='UTC'),
        'temp': 297.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 22:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-29 23:00:00+0000', tz='UTC'),
        'temp': 298.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-29 23:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 00:00:00+0000', tz='UTC'),
        'temp': 299.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 00:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 01:00:00+0000', tz='UTC'),
        'temp': 298.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 01:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 02:00:00+0000', tz='UTC'),
        'temp': 298.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 02:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 03:00:00+0000', tz='UTC'),
        'temp': 297.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 03:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 04:00:00+0000', tz='UTC'),
        'temp': 294.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 04:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 05:00:00+0000', tz='UTC'),
        'temp': 293.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 05:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 06:00:00+0000', tz='UTC'),
        'temp': 292.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 06:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 07:00:00+0000', tz='UTC'),
        'temp': 292.15,
        'temp_unit': 'Kelvin',
    },
    {
        'start_timestamp': pd.Timestamp('2023-08-30 07:00:00+0000', tz='UTC'),
        'end_timestamp': pd.Timestamp('2023-08-30 08:00:00+0000', tz='UTC'),
        'temp': 292.15,
        'temp_unit': 'Kelvin',
    },
]
