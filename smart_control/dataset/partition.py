"""The Building Dataset Partition."""

from functools import cached_property
import os
import pickle

import numpy as np
import pandas as pd

from smart_control.dataset.dataset import BuildingDataset


class BuildingDatasetPartition:
  # pylint:disable=line-too-long
  """A helper class for handling a specific dataset partition.
  A partition is a subset of the building's data over a specific time period.

  The partition contains information about observations, actions, and rewards
  for each time step:

    - **observation**: information the agent receives from the environment
    - **action**: a decision the agent makes to interact with the environment
    - **reward info**: feedback from the environment indicating the agent's
      performance, contains information needed to compute the reward
    - **reward**: results from passing the raw reward info through the reward
      function

  Args:
    dataset (BuildingDataset): The building dataset.
    partition_id (str): The identifier of a partition in the specified dataset
      (e.g. "2022_a").

  Example:
    ```python
    ds = BuildingDataset(dataset_id='sb1', download=True)
    partition = BuildingDatasetPartition(dataset=ds, partition_id='2022_a')
    ```
  """
  # pylint:enable=line-too-long

  def __init__(self, dataset: BuildingDataset, partition_id: str):
    self.ds = dataset
    self.partition_id = partition_id

    if self.partition_id not in self.ds.partition_ids:
      raise ValueError(f"Invalid partition: {self.partition_id}.")

  def __repr__(self):
    return "<BuildingDatasetPartition '{self.ds.dataset_id}':'{self.partition_id}'>"  # pylint:disable=line-too-long

  @property
  def partition_dirpath(self):
    return os.path.join(self.ds.tabular_dirpath, self.ds.dataset_id, self.partition_id)  # pylint:disable=line-too-long

  @property
  def data_filepath(self):
    return os.path.join(self.partition_dirpath, "data.npy.npz")

  @cached_property
  def data(self) -> np.lib.npyio.NpzFile:
    """Time-series data for the dataset partition.

    This property returns an `np.lib.npyio.NpzFile` object, which allows
      dictionary-like access to NumPy arrays stored within a compressed
      `.npz` archive. The arrays contain time-series data, where the first
      dimension typically represents the time steps.

    Returns:
      A dictionary-like numpy object with the following keys:

        - `'observation_value_matrix'`
        - `'action_value_matrix'`
        - `'reward_value_matrix'`
        - `'reward_info_value_matrix'`

        Each of these keys has a corresponding public method for convenience.
        See corresponding documentation below for more information about each.
    """
    return np.load(self.data_filepath)

  @property
  def metadata_filepath(self):
    return os.path.join(self.partition_dirpath, "metadata.pickle")

  @cached_property
  def metadata(self) -> dict:
    """Metadata describing the partition [`data`](./#smart_control.dataset.partition.BuildingDatasetPartition.data).

    Returns:
      A dictionary containing the following keys:

        - `'action_ids_map'`
        - `'action_timestamps'`
        - `'observation_ids'`
        - `'observation_timestamps'`
        - `'reward_info_ids'`
        - `'reward_info_timestamps'`
        - `'reward_timestamps'`

        Each of these keys has a corresponding public method for convenience.
        See corresponding documentation below for more information about each.
    """
    metadata = pickle.load(open(self.metadata_filepath, "rb"))
    # renaming keys:
    metadata = {
        "action_ids_map": metadata["action_ids"],  # renamed
        "action_timestamps": metadata["action_timestamps"],
        "observation_ids_map": metadata["observation_ids"],  # renamed
        "observation_timestamps": metadata["observation_timestamps"],
        "reward_info_ids_map": metadata["reward_ids"],  # renamed
        "reward_timestamps": metadata["reward_timestamps"],
        "reward_info_timestamps": metadata["reward_info_timestamps"],
    }
    return metadata

  #
  # DATA PROPERTIES
  #

  @cached_property
  def action_value_matrix(self) -> np.ndarray:
    """Time-series action data."""
    return self.data["action_value_matrix"]

  @cached_property
  def observation_value_matrix(self) -> np.ndarray:
    """Time-series observation data."""
    return self.data["observation_value_matrix"]

  @cached_property
  def reward_value_matrix(self) -> np.ndarray:
    """Time-series reward data."""
    return self.data["reward_value_matrix"]

  @cached_property
  def reward_info_value_matrix(self) -> np.ndarray:
    """Time series reward information data."""
    return self.data["reward_info_value_matrix"]

  #
  # METADATA PROPERTIES
  #

  @cached_property
  def action_ids_map(self) -> dict:
    """A mapping of unique action identifiers.

    Returns:
      A dictionary where the keys are the [`action_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.action_ids)
        and the values are unique integers referencing column indices in the
        [`action_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.action_value_matrix)

        For example:

        ```py
          {
            '12945159110931775488@supply_air_temperature_setpoint': 0,
            '13761436543392677888@supply_water_temperature_setpoint': 1,
            '14409954889734029312@supply_air_temperature_setpoint': 2
          }
        ```
    """
    return self.metadata["action_ids_map"]

  @cached_property
  def observation_ids_map(self) -> dict:
    """A mapping of unique observation identifiers.

    Returns:
      A dictionary where the keys are the [`observation_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.observation_ids)
        and the values are unique integers referencing column indices in the
        [`observation_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.observation_value_matrix).

        For example:

        ```py
          {
            '202194278473007104@building_air_static_pressure_setpoint', 0,
            ...
            '2640423556868160@zone_air_temperature_sensor': 1197
          }
        ```
    """
    return self.metadata["observation_ids_map"]

  @cached_property
  def reward_info_ids_map(self) -> dict:
    """A mapping of unique reward info identifiers.

    See: `RewardInfo` in "smart_control/proto/smart_control_reward.proto".

    Returns:
      A dictionary where the keys are the [`reward_info_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_info_ids)
        and the values are unique integers referencing column indices in the [`reward_info_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_info_value_matrix).

        For example:

        ```py
          {
            'rooms/9028552126@heating_setpoint_temperature': 0
            ...
            '14409954889734029312@air_conditioning_electrical_energy_rate': 3251
          }
        ```
    """
    return self.metadata["reward_info_ids_map"]

  @cached_property
  def reward_ids_map(self) -> dict:
    """A mapping of unique reward identifiers.

    See: `RewardResponse` in "smart_control/proto/smart_control_reward.proto".

    Returns:
      A dictionary where the keys are the [`reward_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_ids)
        and the values are unique integers referencing column indices in the [`reward_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_value_matrix).
    """
    return {
        "agent_reward_value": 0,
        "productivity_reward": 1,
        "electricity_energy_cost": 2,
        "natural_gas_energy_cost": 3,
        "carbon_emitted": 4,
        "carbon_cost": 5,
        "productivity_weight": 6,
        "energy_cost_weight": 7,
        "carbon_emission_weight": 8,
        "person_productivity": 9,
        "total_occupancy": 10,
        "reward_scale": 11,
        "reward_shift": 12,
        "productivity_regret": 13,
        "normalized_productivity_regret": 14,
        "normalized_energy_cost": 15,
        "normalized_carbon_emission": 16,
    }

  @cached_property
  def action_ids(self) -> list[str]:
    """A list of unique action identifiers.

    Action identifiers are in the format of `device_id@field_name`.
    For example: `'12945159110931775488@supply_air_temperature_setpoint'`.
    """
    return list(self.action_ids_map.keys())

  @cached_property
  def observation_ids(self) -> list[str]:
    """A list of unique observation identifiers.

    Observation identifiers are in the format of `device_id@field_name`.
    For example: `'2640423556868160@zone_air_temperature_sensor'`.
    """
    return list(self.observation_ids_map.keys())

  @cached_property
  def reward_ids(self) -> list[str]:
    """A list of unique reward identifiers.

    See: `RewardResponse` in "smart_control/proto/smart_control_reward.proto".

    Returns:
      A list of the reward identifiers:

        [
          "agent_reward_value",
          "productivity_reward",
          "electricity_energy_cost",
          "natural_gas_energy_cost",
          "carbon_emitted",
          "carbon_cost",
          "productivity_weight",
          "energy_cost_weight",
          "carbon_emission_weight",
          "person_productivity",
          "total_occupancy",
          "reward_scale",
          "reward_shift",
          "productivity_regret",
          "normalized_productivity_regret",
          "normalized_energy_cost",
          "normalized_carbon_emission"
        ]
    """
    return list(self.reward_ids_map.keys())

  @cached_property
  def reward_info_ids(self) -> list[str]:
    """A list of unique reward info identifiers.

    See: `RewardInfo` in "smart_control/proto/smart_control_reward.proto".

    Reward info identifiers are in the format of `device_id@field_name` or
    `zone_id@field_name`.
    For example:

      + `'rooms/9028552126@heating_setpoint_temperature'`
      + `'14409954889734029312@air_conditioning_electrical_energy_rate'`
    """
    return list(self.reward_info_ids_map.keys())

  @cached_property
  def action_timestamps(self) -> list[pd.Timestamp]:
    """A list of sequential timestamps representing the time of each action."""
    return self.metadata["action_timestamps"]

  @cached_property
  def observation_timestamps(self) -> list[pd.Timestamp]:
    """A list of sequential timestamps representing the time of each
    observation.
    """
    return self.metadata["observation_timestamps"]

  @cached_property
  def reward_timestamps(self) -> list[pd.Timestamp]:
    """A list of sequential timestamps representing the time of each reward."""
    return self.metadata["reward_timestamps"]

  @cached_property
  def reward_info_timestamps(self) -> list[pd.Timestamp]:
    """A list of sequential timestamps related to reward information."""
    return self.metadata["reward_info_timestamps"]

  #
  # DATAFRAME PROPERTIES
  #

  def _construct_time_series_df(self, matrix_name, ids_name, timestamps_name):
    """Constructs a dataframe, using matrix values from the partition data,
    as well as column names and index values from the partition metadata.
    """
    # using getattr() to leverage cached properties...
    df = pd.DataFrame(getattr(self, matrix_name))
    columns_map = {v: k for k, v in getattr(self, ids_name).items()}
    df = df.rename(columns=columns_map)
    df.index = getattr(self, timestamps_name)
    df.index.name = "timestamp"
    df.sort_index()  # ensure timestamps are in ascending order
    return df

  @cached_property
  def actions_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A time-series dataframe of numeric action values, constructed from the
    following components:

      + Columns are the [`action_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.action_ids)
      + Row indices are the [`action_timestamps`](./#smart_control.dataset.partition.BuildingDatasetPartition.action_timestamps)
      + Cell values are from the [`action_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.action_value_matrix)

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        | timestamp                 | 12945159110931775488@supply_air_temperature_setpoint  | ... | 14409954889734029312@supply_air_temperature_setpoint  |
        |---------------------------|-------------------------------------------------------|-----|-------------------------------------------------------|
        | 2022-01-01 00:00:00+00:00 | 288.703705                                            | ... | 291.481476                                            |
        | 2022-01-01 00:05:00+00:00 | 288.703705                                            | ... | 291.481476                                            |
        | 2022-01-01 00:10:00+00:00 | 288.703705                                            | ... | 291.481476                                            |
        | 2022-01-01 00:15:00+00:00 | 288.703705                                            | ... | 291.481476                                            |
        | 2022-01-01 00:20:00+00:00 | 288.703705                                            | ... | 291.481476                                            |

    """
    # pylint: enable=line-too-long
    return self._construct_time_series_df(
        matrix_name="action_value_matrix",
        ids_name="action_ids_map",
        timestamps_name="action_timestamps",
    )

  @cached_property
  def observations_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A time-series dataframe of numeric observation values, constructed from the
    following components:

      + Columns are the [`observation_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.observation_ids)
      + Row indices are the [`observation_timestamps`](./#smart_control.dataset.partition.BuildingDatasetPartition.observation_timestamps)
      + Cell values are from the [`observation_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.observation_value_matrix)

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        | timestamp                 | 202194278473007104@building_air_static_pressure_setpoint | ... | 2640423556868160@zone_air_temperature_sensor |
        |---------------------------|----------------------------------------------------------|-----|----------------------------------------------|
        | 2022-01-01 00:00:00+00:00 | 7.472401                                                 | ... | 68.500000                                    |
        | 2022-01-01 00:05:00+00:00 | 7.472401                                                 | ... | 68.300003                                    |
        | 2022-01-01 00:10:00+00:00 | 7.472401                                                 | ... | 68.300003                                    |
        | 2022-01-01 00:15:00+00:00 | 7.472401                                                 | ... | 68.000000                                    |
        | 2022-01-01 00:20:00+00:00 | 7.472401                                                 | ... | 68.000000                                    |

    """
    # pylint: enable=line-too-long
    return self._construct_time_series_df(
        matrix_name="observation_value_matrix",
        ids_name="observation_ids_map",
        timestamps_name="observation_timestamps",
    )

  @cached_property
  def rewards_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A time-series dataframe of numeric reward values, constructed from the
    following components:

      + Columns are the [`reward_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_ids)
      + Row indices are the [`reward_timestamps`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_timestamps)
      + Cell values are from the [`reward_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_value_matrix)

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        | timestamp                 | agent_reward_value | ... | normalized_carbon_emission |
        |---------------------------|--------------------|-----|----------------------------|
        | 2021-12-31 23:55:00+00:00 | -1.005403e-08      | ... | 1.797313e-08               |
        | 2022-01-01 00:00:00+00:00 | -1.002312e-08      | ... | 1.782538e-08               |
        | 2022-01-01 00:05:00+00:00 | -1.002312e-08      | ... | 1.782538e-08               |
        | 2022-01-01 00:10:00+00:00 | -1.002312e-08      | ... | 1.782538e-08               |
        | 2022-01-01 00:15:00+00:00 | -5.737567e-09      | ... | 1.020384e-08               |
    """
    # pylint: enable=line-too-long
    return self._construct_time_series_df(
        matrix_name="reward_value_matrix",
        ids_name="reward_ids_map",
        timestamps_name="reward_timestamps",
    )

  @cached_property
  def reward_infos_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A time-series dataframe of numeric reward info values, constructed from
    the following components:

      + Columns are the [`reward_info_ids`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_info_ids)
      + Row indices are the [`reward_info_timestamps`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_info_timestamps)
      + Cell values are from the [`reward_info_value_matrix`](./#smart_control.dataset.partition.BuildingDatasetPartition.reward_info_value_matrix)

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        | timestamp                 | rooms/9028552126@heating_setpoint_temperature | ... | 14409954889734029312@air_conditioning_electrical_energy_rate |
        |---------------------------|-----------------------------------------------|-----|---------------------------------------------------------------|
        | 2021-12-31 23:55:00+00:00 | 294.0                                         | ... | 0.0                                                           |
        | 2022-01-01 00:00:00+00:00 | 294.0                                         | ... | 0.0                                                           |
        | 2022-01-01 00:05:00+00:00 | 294.0                                         | ... | 0.0                                                           |
        | 2022-01-01 00:10:00+00:00 | 294.0                                         | ... | 0.0                                                           |
        | 2022-01-01 00:15:00+00:00 | 294.0                                         | ... | 0.0                                                           |
    """
    # pylint: enable=line-too-long
    return self._construct_time_series_df(
        matrix_name="reward_info_value_matrix",
        ids_name="reward_info_ids_map",
        timestamps_name="reward_info_timestamps",
    )


if __name__ == "__main__":

  selected_dataset_id = input("Please select a building (e.g. 'sb1'): ") or "sb1"  # pylint:disable=line-too-long
  ds = BuildingDataset(selected_dataset_id, download=True)
  print(ds)

  selected_partition_id = input("Please select a partition (e.g. '2022_a'): ") or "2022_a"  # pylint:disable=line-too-long
  partition = BuildingDatasetPartition(ds, selected_partition_id)
  print(partition)

  actions_df = partition.actions_df
  print("ACTIONS:", actions_df.shape)
  print(actions_df.index[0])
  print(actions_df.index[-1])

  observations_df = partition.observations_df
  print("OBSERVATIONS:", observations_df.shape)
  print(observations_df.index[0])
  print(observations_df.index[-1])

  rewards_df = partition.rewards_df
  print("REWARDS:", rewards_df.shape)
  print(rewards_df.index[0])
  print(rewards_df.index[-1])

  reward_infos_df = partition.reward_infos_df
  print("REWARD INFOS:", reward_infos_df.shape)
  print(reward_infos_df.index[0])
  print(reward_infos_df.index[-1])
