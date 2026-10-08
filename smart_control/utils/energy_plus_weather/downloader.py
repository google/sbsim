"""Downloads and reads EPW weather files from the EnergyPlus archive."""

import argparse
from pathlib import Path
import shutil
from urllib import parse
from urllib import request
import zipfile

import pandas as pd
from pvlib import iotools

ENERGYPLUS_WEATHER_HOST = 'energyplus-weather.s3.amazonaws.com'
MOFFETT_FIELD_TMY3_URL = (
    f'https://{ENERGYPLUS_WEATHER_HOST}/'
    'north_and_central_america_wmo_region_4/USA/CA/'
    'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3/'
    'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip'
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parent / 'test_data'


class Downloader:
  """Downloads and extracts official EnergyPlus weather archives."""

  def __init__(
      self, output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY
  ) -> None:
    """Initializes the downloader.

    Args:
      output_directory: Directory in which downloaded and extracted files are
        written.
    """
    self._output_directory = Path(output_directory)

  def download(self, zip_file_url: str, timeout: int = 30) -> Path:
    """Downloads an EnergyPlus ZIP archive.

    Args:
      zip_file_url: HTTPS URL for a ZIP archive on the official EnergyPlus
        weather host.
      timeout: Number of seconds to wait for the server response.

    Returns:
      The path of the downloaded ZIP archive.

    Raises:
      ValueError: If the URL is not an official EnergyPlus ZIP archive.
    """
    parsed_url = parse.urlparse(zip_file_url)
    if (
        parsed_url.scheme != 'https'
        or parsed_url.hostname != ENERGYPLUS_WEATHER_HOST
        or not parsed_url.path.lower().endswith('.zip')
    ):
      raise ValueError(
          'zip_file_url must be an HTTPS ZIP archive on the official '
          f'EnergyPlus weather host, {ENERGYPLUS_WEATHER_HOST}'
      )

    self._output_directory.mkdir(parents=True, exist_ok=True)
    output_path = self._output_directory / Path(parsed_url.path).name
    with request.urlopen(
        zip_file_url, timeout=timeout
    ) as response:  # nosec B310
      with output_path.open('wb') as output_file:
        shutil.copyfileobj(response, output_file)
    return output_path

  def extract(self, zip_file_path: str | Path) -> Path:
    """Extracts the only EPW file in an EnergyPlus archive.

    Args:
      zip_file_path: Path to a downloaded EnergyPlus ZIP archive.

    Returns:
      The path of the extracted EPW file.

    Raises:
      ValueError: If the archive does not contain exactly one EPW file.
    """
    with zipfile.ZipFile(zip_file_path) as archive:
      epw_members = [
          info
          for info in archive.infolist()
          if not info.is_dir() and info.filename.lower().endswith('.epw')
      ]
      if len(epw_members) != 1:
        raise ValueError(
            'EnergyPlus archive must contain exactly one EPW file; found '
            f'{len(epw_members)}'
        )

      output_path = self._output_directory / Path(epw_members[0].filename).name
      self._output_directory.mkdir(parents=True, exist_ok=True)
      output_path.write_bytes(archive.read(epw_members[0]))
      return output_path


class Reader:
  """Reads EPW files into tabular weather data and site metadata."""

  def read(self, epw_file_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Reads one EPW file with pvlib.

    Args:
      epw_file_path: Path to an extracted EPW weather file.

    Returns:
      A pair containing hourly weather data and site metadata.
    """
    return iotools.read_epw(epw_file_path)


def main() -> None:
  """Downloads and extracts one EnergyPlus weather archive."""
  parser = argparse.ArgumentParser(
      description='Download an EPW file from the EnergyPlus weather archive.'
  )
  parser.add_argument('zip_file_url', help='EnergyPlus weather ZIP URL')
  parser.add_argument(
      '--output-directory',
      type=Path,
      default=DEFAULT_OUTPUT_DIRECTORY,
      help=f'Destination directory (default: {DEFAULT_OUTPUT_DIRECTORY})',
  )
  parser.add_argument(
      '--timeout',
      type=int,
      default=30,
      help='Server response timeout in seconds (default: 30)',
  )
  args = parser.parse_args()

  downloader = Downloader(args.output_directory)
  zip_file_path = downloader.download(args.zip_file_url, args.timeout)
  print(downloader.extract(zip_file_path))


if __name__ == '__main__':
  main()
