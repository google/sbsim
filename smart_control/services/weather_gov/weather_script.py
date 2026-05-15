"""An executable script for integration testing the weather service.

Example execution:
blaze run //third_party/py/smart_buildings/smart_control/services/weather_gov:weather_script
"""  # pylint: disable=line-too-long

from collections.abc import Sequence

from absl import app
from absl import flags

from smart_buildings.smart_control.services.weather_gov import weather_service

_LAT = flags.DEFINE_float(
    name="lat",
    default=weather_service.SB1_COORDS["lat"],
    help=(
        "Latitude (in fractional degrees / decimal degrees) of the location to"
        " fetch weather data for."
    ),
)

_LON = flags.DEFINE_float(
    name="lon",
    default=weather_service.SB1_COORDS["lon"],
    help=(
        "Longitude (in fractional degrees / decimal degrees) of the location to"
        " fetch weather data for."
    ),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  lat = float(input("Provide a latitude (or press enter): ") or _LAT.value)
  lon = float(input("Provide a longitude (or press enter): ") or _LON.value)
  service = weather_service.WeatherService(lat=lat, lon=lon)

  print(f"Fetching forecast for coordinates: ({lat}, {lon})")
  forecast = service.get_forecast()

  print("Current Forecast:")
  print(forecast.df.head())


if __name__ == "__main__":
  app.run(main)
