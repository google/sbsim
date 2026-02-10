"""Smart Buildings Dataset implementation, including loading and downloading."""

from functools import cached_property
import json
import os
import pickle
import shutil

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests

from smart_control.utils.constants import ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

VALID_DATASET_PARTITIONS = {
    "sb1": ["2022_a", "2022_b", "2023_a", "2023_b", "2024_a"]
}
"""Specifies the available partition identifiers for each dataset."""


class BuildingDataset:
  """A helper class for handling the dataset for a specific building.

  Args:
    dataset_id (str): The identifier of the building dataset (e.g. "sb1").
    download (bool): Whether or not to download the dataset.

  Examples:
    ```python
    ds = BuildingDataset(dataset_id="sb1", download=True)
    ```
  """

  def __init__(self, dataset_id="sb1", download=True):
    self.dataset_id = dataset_id

    if self.dataset_id not in VALID_DATASET_PARTITIONS:
      raise ValueError("Invalid building: '{self.dataset_id}'.")

    self.partition_ids = VALID_DATASET_PARTITIONS[self.dataset_id]

    if bool(download):
      self.download()

  def __repr__(self):
    return f"<BuildingDataset '{self.dataset_id}'>"

  @property
  def zip_filename(self):
    """The name of the zip file (e.g. "sb1.zip")."""
    return f"{self.dataset_id}.zip"

  @property
  def zip_url(self):
    """The URL of the zip file located on Google Cloud Storage."""
    return (
        "https://storage.googleapis.com/gresearch/smart_buildings_dataset/"
        f"tabular_data/{self.zip_filename}"
    )

  @property
  def zip_filepath(self):
    """The filepath of the local zip file after it has been downloaded."""
    return os.path.join(DATA_DIR, self.zip_filename)

  @property
  def building_dirpath(self):
    """The local directory containing the building's dataset, after it has been
    extracted from the local zip file.
    """
    return os.path.join(DATA_DIR, self.dataset_id)

  def download(self, timeout=60):
    """Downloads the building's dataset from Google Cloud Storage.

    Only downloads and unzips the dataset if it doesn't already exist at the
      expected [`building_dirpath`](./#smart_control.dataset.dataset.BuildingDataset.building_dirpath)
      location. Otherwise it will load the existing local data.

    Download speed is fairly quick, but unzipping takes a few moments.
    """
    if os.path.isdir(self.building_dirpath):
      print("Using previously-downloaded data...")
      print(os.path.abspath(self.building_dirpath))
    else:
      print("Downloading zip file...")
      print(self.zip_url)
      with requests.get(self.zip_url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(self.zip_filepath, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

      print("Unpacking zip file...")
      print(os.path.abspath(self.zip_filepath))
      shutil.unpack_archive(self.zip_filepath, self.building_dirpath)

  @property
  def tabular_dirpath(self):
    return os.path.join(self.building_dirpath, "tabular")

  # FLOORPLAN

  @property
  def floorplan_filepath(self):
    return os.path.join(self.tabular_dirpath, "floorplan.npy")

  @cached_property
  def floorplan(self) -> np.ndarray:
    """The building's floorplan, as a numpy array.

    The floorplan consists of a map of pixels with the following values:

      + 0: inside / internal space
      + 1: wall / boundary
      + 2: outside / external space

    Use the [`display_floorplan`](./#smart_control.dataset.dataset.BuildingDataset.display_floorplan)
      method to view an image of the floorplan.
    """
    return np.load(self.floorplan_filepath)

  @property
  def floorplan_image_filepath(self):
    """Filepath for saving an image of the floorplan."""
    floorplan_image_filename = f"{self.dataset_id}_floorplan.png"
    return os.path.join(DOCS_DIR, "assets", "images", floorplan_image_filename)

  def display_floorplan(
      self,
      cmap="binary",
      show=True,
      save=True,
      image_filepath: str | None = None,
  ):
    """Renders an image of the building's floorplan.

    Here is an example floorplan for building "sb1":

    ![An image of a floorplan.](../../assets/images/sb1_floorplan.png)

    Args:
      cmap (str): The name of a [matplotlib color map](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        to use when rendering the image.
      show (bool): Whether or not to show the image.
      save (bool): Whether or not to save the image (as a .png file).
      image_filepath (str): An optional custom filepath to use when saving the
        image. Only applies if `save=True`. By default, saves to the [`floorplan_image_filepath`](./#smart_control.dataset.dataset.BuildingDataset.floorplan_image_filepath)
    """
    plt.imshow(self.floorplan, interpolation="nearest", cmap=cmap)
    if show:
      plt.show()
    if save:
      image_filepath = image_filepath or self.floorplan_image_filepath
      plt.savefig(image_filepath)

  # DEVICES

  @property
  def device_layout_map_filepath(self):
    return os.path.join(self.tabular_dirpath, "device_layout_map.json")

  @cached_property
  def device_layout_map(self) -> dict:
    """A layout map of devices in the building.

    Returns:
      A dictionary with keys corresponding to each of the device layouts
        (e.g. 'VAV CO 1-1-06'). Each value is a list of integer coordinates.
        The length of the coordinates list is not the same across all devices.
        Here is an abbreviated version of the device layout map:

        ```py
        {
          'VAV CO 1-1-06': [[79, 35], [80, 35], [80, 34], [80, 33], ... ],
          ...
          'VAV RH 1-1-28 CO2 (Tech Talk 1H2)': [[22, 422], [23, 422], ... ],
          ...
          'VAV RH 1-1-55': [[145, 126], [146, 126], [147, 126], ... ]
        }
        ```
    """
    with open(self.device_layout_map_filepath, encoding="utf-8") as json_file:
      return json.load(json_file)

  @property
  def device_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "device_info_dicts.pickle")

  @cached_property
  def device_infos(self) -> list[dict]:
    """Information about the devices in the building.

    Returns:
      A list of device dictionaries. Here is an example device dictionary:

        ```py
        {
            'device_id': '202194278473007104',
            'namespace': 'PHRED',
            'code': 'SB1:AHU:AC-2',
            'zone_id': '',
            'device_type': 6,
            'observable_fields': {
                'building_air_static_pressure_sensor': 1,
                'outside_air_flowrate_sensor': 1,
                'supply_fan_speed_percentage_command': 1,
                'supply_air_temperature_sensor': 1,
                'supply_fan_speed_frequency_sensor': 1,
                'supply_air_static_pressure_setpoint': 1,
                'return_air_temperature_sensor': 1,
                'mixed_air_temperature_setpoint': 1,
                'exhaust_fan_speed_percentage_command': 1,
                'exhaust_fan_speed_frequency_sensor': 1,
                'outside_air_damper_percentage_command': 1,
                'mixed_air_temperature_sensor': 1,
                'exhaust_air_damper_percentage_command': 1,
                'cooling_percentage_command': 1,
                'outside_air_flowrate_setpoint': 1,
                'supply_air_temperature_setpoint': 1,
                'building_air_static_pressure_setpoint': 1,
                'supply_air_static_pressure_sensor': 1,
            },
            'actionable_fields': {
                'exhaust_air_damper_percentage_command': 1,
                'supply_air_temperature_setpoint': 1,
                'supply_fan_speed_percentage_command': 1,
                'outside_air_flowrate_setpoint': 1,
                'cooling_percentage_command': 1,
                'mixed_air_temperature_setpoint': 1,
                'exhaust_fan_speed_percentage_command': 1,
                'outside_air_damper_percentage_command': 1,
                'supply_air_static_pressure_setpoint': 1,
                'building_air_static_pressure_setpoint': 1,
            },
        }
        ```
    """
    infos = pickle.load(open(self.device_infos_filepath, "rb"))
    for info in infos:
      if "action_fields" in info:
        info["actionable_fields"] = info.pop("action_fields")  # rename
    return infos

  @cached_property
  def devices_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A dataframe containing information about the building's devices.

    Each row is uniquely identified by the "device_id".

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        |    |          device_id | namespace   | code              |   device_type | observable_fields                                     | actionable_fields                                                                                                                                                                                                                                                                                                                                                                                                       |
        |---:|-------------------:|:------------|:------------------|--------------:|:------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        |  0 | 202194278473007104 | PHRED       | SB1:AHU:AC-2      |             6 | {'building_air_static_pressure_sensor': 1,  ... }     | {'exhaust_air_damper_percentage_command': 1, ... }
        |  1 |   2760348383893915 | CDM         | VAV CO 1-1-10 CO2 |             4 | {'zone_air_heating_temperature_setpoint': 1, ... }    | {'supply_air_damper_percentage_command': 1, ... }                                                                                                                                                                                                    |
        |  2 |   2562701969438717 | CDM         | VAV CO 2-2-36     |             4 | {'zone_air_heating_temperature_setpoint': 1, ... }    | {'supply_air_flowrate_setpoint': 1, ... }                                                                                                                                                                                                                                              |
        |  3 |   2806035809406684 | CDM         |                   |             4 | {'discharge_air_temperature_setpoint': 1, ... }       | {'discharge_air_temperature_setpoint': 1, ... }                                                                                                                                                        |
        |  4 |   2790439929052995 | CDM         | VAV CO 1-1-43     |             4 | {'zone_air_heating_temperature_setpoint': 1, ... }    | {'supply_air_damper_percentage_command': 1, ... }                                                                                                                                                                                                                                              |
    """
    # pylint: enable=line-too-long
    df = pd.DataFrame(self.device_infos)
    df = df.drop(columns=["zone_id"])  # many to many relationship
    # consider renaming zone_id to zone_ids instead of dropping
    return df

  # DEVICE FIELDS

  def _count_device_fields(self, field_type: str) -> pd.Series:
    """
    Param field_type (str) member of: ["actionable_fields", "observable_fields"]
    """
    field_counts = {}

    for device_info in self.device_infos:
      device_fields = list(device_info[field_type].keys())

      for field in device_fields:
        if field in field_counts:
          field_counts[field] += 1
        else:
          field_counts[field] = 1

    value_counts = pd.Series(field_counts).sort_values(ascending=False)
    value_counts.index.name = "field_name"
    return value_counts

  @cached_property
  def actionable_field_counts(self) -> pd.Series:
    """Counts the number of devices supporting each actionable field."""
    return self._count_device_fields("actionable_fields")

  @cached_property
  def actionable_fields(self) -> list[str]:
    """Names of all unique actionable fields across all devices."""
    return sorted(self.actionable_field_counts.keys())

  @cached_property
  def observable_field_counts(self) -> pd.Series:
    """Counts the number of devices supporting each observable field."""
    return self._count_device_fields("observable_fields")

  @cached_property
  def observable_fields(self) -> list[str]:
    """Names of all unique observable fields across all devices."""
    return sorted(self.observable_field_counts.keys())

  @cached_property
  def fields_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A dataframe containing information about all device fields in the
    building, including whether each is observable and/or actionable.

    Each row is uniquely identified by the "field_name".

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        |    | field_name                            | is_actionable | is_observable | n_devices_actionable | n_devices_observable |
        |---:|:--------------------------------------|:--------------|:--------------|:---------------------|:---------------------|
        |  0 | building_air_static_pressure_sensor   | False         | True          | 0                    | 3                    |
        |  1 | building_air_static_pressure_setpoint | True          | True          | 3                    | 3                    |
        |  2 | cooling_percentage_command            | True          | True          | 3                    | 3                    |
        |  3 | differential_pressure_sensor          | False         | True          | 0                    | 2                    |
        |  4 | differential_pressure_setpoint        | True          | True          | 2                    | 2                    |

    """
    # pylint: enable=line-too-long
    actionable_fields = set(self.actionable_fields)
    observable_fields = set(self.observable_fields)
    all_fields = actionable_fields.union(observable_fields)

    records = []
    for field in all_fields:
      records.append({
          "field_name": field,
          "is_actionable": field in actionable_fields,
          "is_observable": field in observable_fields,
          "n_devices_actionable": self.actionable_field_counts.get(field, 0),
          "n_devices_observable": self.observable_field_counts.get(field, 0),
      })
    return pd.DataFrame(records).sort_values(by="field_name")

  # ZONES

  @property
  def zone_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "zone_info_dicts.pickle")

  @cached_property
  def zone_infos(self) -> list[dict]:
    """Information about the zones in the building.

    Returns:
      A list of zone dictionaries. Here is an example zone dictionary:

        ```py
        {
          'zone_id': 'rooms/1002000133978',
          'building_id': 'buildings/3616672508',
          'zone_description': 'SB1-2-C2054',
          'area': 0.0,
          'zone_type': 1,
          'floor': 2,
          'devices': ['2618581107144046', '2696593986887004'],
        }
        ```
    """
    return pickle.load(open(self.zone_infos_filepath, "rb"))

  @cached_property
  def zones_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A dataframe containing information about the building's zones.

    Each row is uniquely identified by the "zone_id".

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        |    | zone_id             | building_id          | zone_description   |   area |   zone_type |   floor | devices                                  |   n_devices |
        |---:|:--------------------|:---------------------|:-------------------|-------:|------------:|--------:|:-----------------------------------------|------------:|
        |  0 | rooms/1002000133978 | buildings/3616672508 | SB1-2-C2054        |      0 |           1 |       2 | ['2618581107144046', '2696593986887004'] |           2 |
        |  1 | rooms/9028471695    | buildings/3616672508 | SB1-2-2D4A         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  2 | rooms/9028472496    | buildings/3616672508 | SB1-2-2D4H         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  3 | rooms/9028558963    | buildings/3616672508 | SB1-2-2D4B         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  4 | rooms/9028483453    | buildings/3616672508 | SB1-2-2D4G         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
    """
    # pylint: enable=line-too-long
    df = pd.DataFrame(self.zone_infos)
    df["n_devices"] = df["devices"].apply(len)
    return df


if __name__ == "__main__":

  ds = BuildingDataset()

  # save building image to docs directory:
  ds.display_floorplan(show=False, save=True)
