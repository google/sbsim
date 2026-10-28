# EnergyPlus weather data

Solar-radiation simulations can use EnergyPlus Weather (EPW) files directly. EPW
preserves the original EnergyPlus fields and can be loaded with
`pvlib.iotools.read_epw`; converting it to CSV does not turn it into the
different TMY3 CSV format expected by `pvlib.iotools.read_tmy3`.

Use the downloader with a ZIP URL from the official EnergyPlus weather archive:

```sh
python -m smart_control.utils.energy_plus_weather \
  https://energyplus-weather.s3.amazonaws.com/north_and_central_america_wmo_region_4/USA/CA/USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3/USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip
```

By default, the command writes downloaded and extracted files to
`smart_control/utils/energy_plus_weather/test_data`. Use `--output-directory` to
select a different destination.

The same workflow is available programmatically:

```python
from smart_control.utils.energy_plus_weather import Downloader, Reader
from smart_control.utils.energy_plus_weather.downloader import MOFFETT_FIELD_TMY3_URL

downloader = Downloader('/tmp/weather')
zip_file_path = downloader.download(MOFFETT_FIELD_TMY3_URL)
epw_file_path = downloader.extract(zip_file_path)
weather_data, metadata = Reader().read(epw_file_path)
```

The Moffett Field example is a typical meteorological year, not observations
from a particular calendar year. Use the existing timestamped SB-1 weather files
when a simulation requires recorded weather.

## API reference

::: smart_control.utils.energy_plus_weather.downloader
