# Weather data

Solar-radiation simulations can use EnergyPlus Weather (EPW) files directly.
EPW preserves the original EnergyPlus fields and can be loaded with
`pvlib.iotools.read_epw`; converting it to a CSV does not turn it into the
different TMY3 CSV format expected by `pvlib.iotools.read_tmy3`.

Use the downloader with a ZIP URL from the official EnergyPlus weather archive:

```sh
python -m smart_control.utils.energyplus_weather \
  https://energyplus-weather.s3.amazonaws.com/north_and_central_america_wmo_region_4/USA/CA/USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3/USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.zip \
  --output-directory smart_control/configs/resources/sb1/weather_data
```

The command validates the official archive host and extracts the archive's one
EPW file. Load the result without an intermediate format conversion:

```python
from pvlib.iotools import read_epw

weather, metadata = read_epw(
    'smart_control/configs/resources/sb1/weather_data/'
    'USA_CA_Mountain.View-Moffett.Field.NAS.745090_TMY3.epw'
)
```

The Moffett Field example is a typical meteorological year, not observations
from a particular calendar year. Use the existing timestamped files in the same
resource directory when a simulation requires recorded SB-1 weather instead.
