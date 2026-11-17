"""Tests for the EnergyPlus weather downloader and reader."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from smart_control.utils.energy_plus_weather import downloader

_TEST_DATA_DIRECTORY = Path(__file__).parent / 'test_data'
_MOFFETT_FIELD_ARCHIVE = (
    _TEST_DATA_DIRECTORY
    / 'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip'
)


def _archive(*members: tuple[str, bytes]) -> bytes:
  output = io.BytesIO()
  with zipfile.ZipFile(output, 'w') as archive:
    for name, content in members:
      archive.writestr(name, content)
  return output.getvalue()


class DownloaderTest(unittest.TestCase):

  @mock.patch.object(downloader.request, 'urlopen')
  def test_download_writes_archive_and_uses_timeout(self, urlopen):
    archive_bytes = _MOFFETT_FIELD_ARCHIVE.read_bytes()
    urlopen.return_value = io.BytesIO(archive_bytes)

    with tempfile.TemporaryDirectory() as output_directory:
      zip_file_path = downloader.Downloader(output_directory).download(
          downloader.MOFFETT_FIELD_TMY3_URL, timeout=17
      )

      self.assertEqual(zip_file_path.read_bytes(), archive_bytes)
      urlopen.assert_called_once_with(
          downloader.MOFFETT_FIELD_TMY3_URL, timeout=17
      )

  def test_download_rejects_non_energyplus_url(self):
    with tempfile.TemporaryDirectory() as output_directory:
      with self.assertRaisesRegex(ValueError, 'official EnergyPlus'):
        downloader.Downloader(output_directory).download(
            'https://example.com/weather.zip'
        )

  def test_extract_reads_local_moffett_field_archive(self):
    with tempfile.TemporaryDirectory() as output_directory:
      epw_file_path = downloader.Downloader(output_directory).extract(
          _MOFFETT_FIELD_ARCHIVE
      )

      self.assertEqual(
          epw_file_path.name,
          'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.epw',
      )
      self.assertTrue(epw_file_path.read_text().startswith('LOCATION,'))

  def test_extract_requires_exactly_one_epw_file(self):
    with tempfile.TemporaryDirectory() as output_directory:
      archive_path = Path(output_directory) / 'weather.zip'
      archive_path.write_bytes(
          _archive(('first.epw', b'first'), ('second.epw', b'second'))
      )

      with self.assertRaisesRegex(ValueError, 'exactly one EPW file; found 2'):
        downloader.Downloader(output_directory).extract(archive_path)


class ReaderTest(unittest.TestCase):

  def test_read_returns_usable_moffett_field_data(self):
    with tempfile.TemporaryDirectory() as output_directory:
      weather_downloader = downloader.Downloader(output_directory)
      epw_file_path = weather_downloader.extract(_MOFFETT_FIELD_ARCHIVE)

      weather_data, metadata = downloader.Reader().read(epw_file_path)

      self.assertEqual(len(weather_data), 8760)
      self.assertEqual(metadata['city'], 'Mountain View Moffett Fld Nas')
      self.assertAlmostEqual(metadata['latitude'], 37.4)
      self.assertAlmostEqual(metadata['longitude'], -122.05)
      self.assertIn('temp_air', weather_data.columns)
      self.assertIn('ghi', weather_data.columns)
      self.assertFalse(weather_data[['temp_air', 'ghi']].isna().all().any())


if __name__ == '__main__':
  unittest.main()
