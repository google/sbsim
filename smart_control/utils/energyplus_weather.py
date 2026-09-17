# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downloads EPW weather files from the EnergyPlus weather archive."""

import argparse
import io
from pathlib import Path
from urllib import parse
from urllib import request
import zipfile


ENERGYPLUS_WEATHER_HOST = 'energyplus-weather.s3.amazonaws.com'
MOFFETT_FIELD_TMY3_URL = (
    'https://energyplus-weather.s3.amazonaws.com/'
    'north_and_central_america_wmo_region_4/USA/CA/'
    'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3/'
    'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip'
)


def download_epw(weather_url: str, output_directory: str | Path) -> Path:
  """Downloads one EnergyPlus archive and extracts its EPW file.

  Args:
    weather_url: HTTPS URL for a ZIP archive on the official EnergyPlus
      weather host.
    output_directory: Directory in which to write the EPW file.

  Returns:
    The path of the extracted EPW file.

  Raises:
    ValueError: If the URL is not an official EnergyPlus weather archive or the
      archive does not contain exactly one EPW file.
  """
  parsed_url = parse.urlparse(weather_url)
  if (
      parsed_url.scheme != 'https'
      or parsed_url.hostname != ENERGYPLUS_WEATHER_HOST
      or not parsed_url.path.lower().endswith('.zip')
  ):
    raise ValueError(
        'weather_url must be an HTTPS ZIP archive on the official EnergyPlus '
        'weather host, '
        f'{ENERGYPLUS_WEATHER_HOST}'
    )

  with request.urlopen(weather_url, timeout=30) as response:  # nosec B310
    archive_bytes = response.read()

  with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
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

    output_path = Path(output_directory) / Path(epw_members[0].filename).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive.read(epw_members[0]))
    return output_path


def main() -> None:
  """Runs the EnergyPlus weather downloader."""
  parser = argparse.ArgumentParser(
      description='Download an EPW file from the EnergyPlus weather archive.'
  )
  parser.add_argument('weather_url', help='EnergyPlus weather ZIP URL')
  parser.add_argument(
      '--output-directory',
      type=Path,
      default=Path.cwd(),
      help='Destination directory (default: current directory)',
  )
  args = parser.parse_args()
  print(download_epw(args.weather_url, args.output_directory))


if __name__ == '__main__':
  main()
