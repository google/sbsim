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
"""Tests for energyplus_weather."""

import io
import tempfile
import unittest
from unittest import mock
import zipfile

from smart_control.utils import energyplus_weather


def _archive(*members: tuple[str, bytes]) -> bytes:
  output = io.BytesIO()
  with zipfile.ZipFile(output, 'w') as archive:
    for name, content in members:
      archive.writestr(name, content)
  return output.getvalue()


class EnergyplusWeatherTest(unittest.TestCase):

  @mock.patch.object(energyplus_weather.request, 'urlopen')
  def test_download_epw_extracts_only_weather_file(self, urlopen):
    urlopen.return_value = io.BytesIO(
        _archive(
            ('nested/location.epw', b'LOCATION,Mountain View'),
            ('nested/location.stat', b'statistics'),
        )
    )

    with tempfile.TemporaryDirectory() as output_directory:
      output_path = energyplus_weather.download_epw(
          energyplus_weather.MOFFETT_FIELD_TMY3_URL, output_directory
      )

      self.assertEqual(output_path.name, 'location.epw')
      self.assertEqual(output_path.read_bytes(), b'LOCATION,Mountain View')

  def test_download_epw_rejects_non_energyplus_url(self):
    with tempfile.TemporaryDirectory() as output_directory:
      with self.assertRaisesRegex(ValueError, 'official EnergyPlus'):
        energyplus_weather.download_epw(
            'https://example.com/weather.zip', output_directory
        )

  @mock.patch.object(energyplus_weather.request, 'urlopen')
  def test_download_epw_requires_exactly_one_epw_file(self, urlopen):
    urlopen.return_value = io.BytesIO(
        _archive(('first.epw', b'first'), ('second.epw', b'second'))
    )

    with tempfile.TemporaryDirectory() as output_directory:
      with self.assertRaisesRegex(ValueError, 'exactly one EPW file; found 2'):
        energyplus_weather.download_epw(
            energyplus_weather.MOFFETT_FIELD_TMY3_URL, output_directory
        )


if __name__ == '__main__':
  unittest.main()
