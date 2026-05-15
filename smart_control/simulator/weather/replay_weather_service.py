"""Weather service using replay weather data, for use in simulation."""

import dataclasses

import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast
from smart_buildings.smart_control.services.weather import base_forecast_period
from smart_buildings.smart_control.services.weather import base_weather_service
from smart_buildings.smart_control.simulator import weather_controller
from smart_buildings.smart_control.utils import temperature_conversion

ReplayWeatherController = weather_controller.ReplayWeatherController


@dataclasses.dataclass(frozen=True)
class ReplayHourlyForecastPeriod(base_forecast_period.BaseForecastPeriod):
  """Hourly forecast period returned by the ReplayWeatherService."""


@dataclasses.dataclass(frozen=True)
class ReplayHourlyForecast(base_forecast.BaseForecast):
  """Hourly forecast returned by the ReplayWeatherService."""


class ReplayWeatherService(base_weather_service.BaseWeatherService):
  """Weather forecast service for use in simulation.

  Uses the ReplayWeatherController to fetch historical weather data, which is
  treated as the upcoming forecast (from the perspective of the current
  simulation time).
  """

  def __init__(self, controller: ReplayWeatherController):
    self._controller = controller

  def get_hourly_forecast(
      self,
      start_timestamp: pd.Timestamp | None = None,
      n_hours: int = 24,
      temp_unit: str = "K",
      normalize_to_start: bool = False,
  ) -> ReplayHourlyForecast:
    """Fetches hourly forecast data starting at the given timestamp.

    FYI: Although the replay temperatures are provided in Fahrenheit in the
      CSV files, the ReplayWeatherController converts them to Kelvin.

    Args:
      start_timestamp: The timestamp to start fetching forecast data from.
      n_hours: The number of hours to fetch forecast data for.
      temp_unit: The unit of temperatures to be returned (e.g. "Fahrenheit",
        "Celsius", or "Kelvin"), or just the first letter. Temperatures will be
        converted from Kelvin as necessary.
      normalize_to_start: If False (default), returns forecasts on the hour
        (e.g. 10:00, 11:00, etc.), or whatever was returned by the service. If
        True and start_timestamp is provided, returns forecasts normalized to
        the start_timestamp (e.g. 10:05, 11:05, etc.). Uses linear interpolation
        to calculate forecast values at the normalized times.

    Returns:
      A ReplayHourlyForecast object containing the specified number of periods.
    """
    if start_timestamp is None:
      raise ValueError(
          "start_timestamp must be provided for ReplayWeatherService."
      )

    if not normalize_to_start:
      # Provide forecast periods "on the hour", starting with the current hour:
      start_timestamp = start_timestamp.floor(freq="H")

    periods = []
    display_unit = temperature_conversion.assign_temp_unit(temp_unit)
    for i in range(n_hours):
      current_time = start_timestamp + pd.Timedelta(hours=i)
      temp_k = self._controller.get_current_temp(current_time)  # in Kelvin
      periods.append(
          ReplayHourlyForecastPeriod(
              start_timestamp=current_time,
              end_timestamp=current_time + pd.Timedelta(hours=1),
              temp=temperature_conversion.from_kelvin(temp_k, display_unit),
              temp_unit=display_unit,
          )
      )
    return ReplayHourlyForecast(periods=periods)
