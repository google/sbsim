"""(Even more) Test helpers for the environment module."""

import numpy as np
from tf_agents.specs import array_spec

from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import environment_test_utils
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.proto import smart_control_normalization_pb2
from smart_buildings.smart_control.utils import bounded_action_normalizer
from smart_buildings.smart_control.utils import observation_normalizer

ContinuousVariableInfo = smart_control_normalization_pb2.ContinuousVariableInfo

BoundedActionNormalizer = bounded_action_normalizer.BoundedActionNormalizer
StandardScoreObservationNormalizer = observation_normalizer.StandardScoreObservationNormalizer  # pylint: disable=line-too-long


LAYOUT = {
    "zone_1": {
        "air_handler_1": [
            "setpoint_1",
            "measurement_1",
            "supply_air_heating_temperature_setpoint",
            "supervisor_run_command",
        ],
        "boiler_1": [
            "setpoint_2",
            "setpoint_3",
            "setpoint_4",
            "measurement_2",
            "supply_water_setpoint",
            "supervisor_run_command",
        ],
    },
    "zone_2": {
        "air_handler_2": [
            "measurement_3",
            "supply_air_heating_temperature_setpoint",
            "supervisor_run_command",
        ],
        "vav_1": ["setpoint_5", "measurement_4"],
    },
}


DEMO_LAYOUT = {
    "zone_1": {
        "air_handler_1": [
            "supply_air_heating_temperature_setpoint",
            "supervisor_run_command",
        ],
        "boiler_1": [
            "supply_water_setpoint",
            "supervisor_run_command",
        ],
    },
    "zone_2": {
        "air_handler_2": [
            "supply_air_heating_temperature_setpoint",
            "supervisor_run_command",
        ],
        "outside_air_sensor": ["outside_air_temperature_sensor"],
    },
}


SIM_NEW_HVAC_LAYOUT = {
    "zone_1": {
        "ahu": [
            "ahu_1_supply_air_heating_temperature_setpoint",
            "ahu_2_supply_air_heating_temperature_setpoint",
            "ahu_1_static_pressure_setpoint",
            "ahu_2_static_pressure_setpoint",
            "supervisor_run_command",
        ],
        "hws": [
            "supply_water_setpoint",
            "differential_pressure",
            "supervisor_run_command",
        ],
    },
    "zone_2": {
        "ahu": [
            "ahu_2_supply_air_heating_temperature_setpoint",
            "ahu_2_supply_air_heating_temperature_setpoint",
            "ahu_2_supervisor_run_command",
        ],
        "outside_air_sensor": ["outside_air_temperature_sensor"],
    },
}


OBSERVATION_NORMALIZERS = {
    "temperature": {"sample_mean": 310.0, "sample_variance": 50 * 50},
    "supply_water_setpoint": {
        "sample_mean": 310.0,
        "sample_variance": 50 * 50,
    },
    "air_flowrate": {"sample_mean": 0.5, "sample_variance": 4.0},
    "differential_pressure": {
        "sample_mean": 20000.0,
        "sample_variance": 100000.0,
    },
    "percentage": {"sample_mean": 0.5, "sample_variance": 1.0},
    "request_count": {"sample_mean": 9, "sample_variance": 25.0},
    # ...
    "outside_air_temperature_sensor": {
        "sample_mean": 291.244931,
        "sample_variance": 12.904175,
    },
    "outside_air_relative_humidity_sensor": {
        "sample_mean": 71.799372,
        "sample_variance": 172.388773,
    },
    # ...
    "supply_air_heating_temperature_setpoint": {
        "sample_mean": 289.329414,
        "sample_variance": 3.186769,
    },
    # ...
    "supervisor_run_command": {
        "sample_mean": 0.0,
        "sample_variance": 1,
    },
}

ACTION_NORMALIZERS = {
    "supply_air_heating_temperature_setpoint": {
        "min_native_value": 285.0,
        "max_native_value": 295.0,  # changed from 300 to make the math easier
    },
    "supply_water_setpoint": {
        "min_native_value": 310.0,
        "max_native_value": 350.0,  # changed from 355 to make the math easier
    },
}

HYBRID_ACTION_NORMALIZERS = {
    **ACTION_NORMALIZERS,
    **{
        "supervisor_run_command": {
            "min_native_value": 0,
            "max_native_value": 1,
        },
    },
}

DEVICE_ACTION_TUPLES = [
    ("air_handler_1", "supply_air_heating_temperature_setpoint"),
    ("boiler_1", "supply_water_setpoint"),
    ("air_handler_2", "supply_air_heating_temperature_setpoint"),
]

HYBRID_DEVICE_ACTION_TUPLES = [
    ("air_handler_1", "supply_air_heating_temperature_setpoint"),  # continuous
    ("air_handler_1", "supervisor_run_command"),  # discrete
    ("boiler_1", "supply_water_setpoint"),  # continuous
    ("boiler_1", "supervisor_run_command"),  # discrete
    ("air_handler_2", "supply_air_heating_temperature_setpoint"),  # continuous
    ("air_handler_2", "supervisor_run_command"),  # discrete
]

DEFAULT_ACTIONS = {
    "air_handler_1_supply_air_heating_temperature_setpoint": 290.0,
    "boiler_1_supply_water_setpoint": 310.0,
    "air_handler_2_supply_air_heating_temperature_setpoint": 290.0,
}

DEFAULT_HYBRID_ACTIONS = {
    "air_handler_1_supply_air_heating_temperature_setpoint": 290.0,
    "air_handler_1_supervisor_run_command": 0,
    "boiler_1_supply_water_setpoint": 310.0,
    "boiler_1_supervisor_run_command": 0,
    "air_handler_2_supply_air_heating_temperature_setpoint": 290.0,
    "air_handler_2_supervisor_run_command": 0,
}


def create_building(layout=None, initial_values=None, start_timestamp=None):
  """Building implementation for unit tests."""
  layout = layout or LAYOUT
  initial_values = initial_values or {"outside_air_temperature_sensor": 295.0}
  return environment_test_utils.SimpleBuilding(
      layout=layout,
      initial_values=initial_values,
      start_timestamp=start_timestamp,
  )


def create_observation_normalizer(
    mapping=None,
) -> StandardScoreObservationNormalizer:
  """Creates an observation normalizer to use for testing purposes.

  Args:
    mapping: A dictionary of continuous variable mappings. The keys are the
      normalizer identifiers, and the values are a dictionaries of statistics.

  Returns:
    A StandardScoreObservationNormalizer instance.
  """
  mapping = mapping or OBSERVATION_NORMALIZERS
  mapping = {k: ContinuousVariableInfo(**v) for k, v in mapping.items()}
  return StandardScoreObservationNormalizer(mapping)


def create_action_config(mapping=None):
  """Creates a bounded action config to use for testing purposes.

  Args:
    mapping: A dictionary of action mappings, for creating action normalizers.
      The keys are the action names, and the values are dictionaries of minimum
      and maximum native values.

  Returns:
    An ActionConfig instance.
  """
  mapping = mapping or ACTION_NORMALIZERS
  normalizers = {k: BoundedActionNormalizer(**v) for k, v in mapping.items()}
  return environment.ActionConfig(normalizers)


def create_hybrid_action_config(mapping=None):
  mapping = mapping or HYBRID_ACTION_NORMALIZERS
  return create_action_config(mapping)


def create_environment(
    layout=None,
    device_action_tuples=None,
    observation_normalizers=None,
    action_normalizers=None,
    metrics_path=None,
    writer_factory=None,
    default_actions=None,
    building=None,
):
  """Creates an environment to use for testing purposes."""

  building = building or create_building(layout=layout)
  reward_function = environment_test_utils.SimpleRewardFunction()
  obs_normalizer = create_observation_normalizer(observation_normalizers)
  action_config = create_action_config(action_normalizers)
  device_action_tuples = device_action_tuples or DEVICE_ACTION_TUPLES
  env = environment.Environment(
      building=building,
      reward_function=reward_function,
      observation_normalizer=obs_normalizer,
      action_config=action_config,
      device_action_tuples=device_action_tuples,
      metrics_path=metrics_path,
      writer_factory=writer_factory,
      default_actions=default_actions,
  )
  env.reset()
  return env


def create_hybrid_action_environment(
    layout=None,
    device_action_tuples=None,
    observation_normalizers=None,
    action_normalizers=None,
    metrics_path=None,
    writer_factory=None,
    default_actions=None,
    building=None,
):
  """Creates an environment to use for testing purposes."""

  building = building or create_building(layout=layout)
  reward_function = environment_test_utils.SimpleRewardFunction()
  obs_normalizer = create_observation_normalizer(observation_normalizers)
  action_config = create_hybrid_action_config(action_normalizers)
  device_action_tuples = device_action_tuples or HYBRID_DEVICE_ACTION_TUPLES
  env = hybrid_action_environment.HybridActionEnvironment(
      building=building,
      reward_function=reward_function,
      observation_normalizer=obs_normalizer,
      action_config=action_config,
      device_action_tuples=device_action_tuples,
      metrics_path=metrics_path,
      writer_factory=writer_factory,
      default_actions=default_actions,
  )
  env.reset()
  return env


def create_observation_spec(n_observations: int) -> array_spec.ArraySpec:
  return array_spec.ArraySpec(
      shape=(n_observations,),
      dtype=np.float32,
      name="observation"
  )


def create_action_spec(n_continuous: int) -> array_spec.BoundedArraySpec:
  return array_spec.BoundedArraySpec(
      shape=(n_continuous,),
      dtype=np.float32,
      minimum=-1,
      maximum=1,
      name="action",
  )


def create_hybrid_action_spec(
    n_discrete: int, n_continuous: int
) -> dict[str, array_spec.BoundedArraySpec]:
  return {
      "discrete_action": array_spec.BoundedArraySpec(
          (n_discrete,),
          np.int32,
          minimum=0,
          maximum=1,
          name="discrete_action",
      ),
      "continuous_action": array_spec.BoundedArraySpec(
          (n_continuous,),
          np.float32,
          minimum=-1,
          maximum=1,
          name="continuous_action"
      ),
  }
