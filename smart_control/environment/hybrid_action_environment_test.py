"""Tests for hybrid action environment."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# pylint: disable=g-bad-import-order we prefer local imports below packages
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import environment_test_utils
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.models import base_building
from smart_buildings.smart_control.models import base_reward_function
from smart_buildings.smart_control.proto import smart_control_normalization_pb2
from smart_buildings.smart_control.utils import bounded_action_normalizer
from smart_buildings.smart_control.utils import observation_normalizer

SimpleBuildingHybridAction = environment_test_utils.SimpleBuildingHybridAction


class HybridActionEnvironmentTest(parameterized.TestCase, tf.test.TestCase):

  def _create_observation_normalizer(self):
    normalization_constants = {
        "temperature": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="temperature", sample_mean=310.0, sample_variance=50 * 50
        ),
        "supply_water_setpoint": (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id="supply_water_setpoint",
                sample_mean=310.0,
                sample_variance=50 * 50,
            )
        ),
        "air_flowrate": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="air_flowrate", sample_mean=0.5, sample_variance=4.0
        ),
        "differential_pressure": (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id="differential_pressure",
                sample_mean=20000.0,
                sample_variance=100000.0,
            )
        ),
        "percentage": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="percentage", sample_mean=0.5, sample_variance=1.0
        ),
        "request_count": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="request_count", sample_mean=9, sample_variance=25.0
        ),
        "measurement": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="measurement", sample_mean=0.0, sample_variance=1.0
        ),
    }

    return observation_normalizer.StandardScoreObservationNormalizer(
        normalization_constants
    )

  def _create_bounded_action_config(self, min_value, max_value):
    action_normalizer = bounded_action_normalizer.BoundedActionNormalizer(
        min_value, max_value
    )
    run_command_normalizer = bounded_action_normalizer.BoundedActionNormalizer(
        -1.0, 1.0
    )

    action_normalizer_inits = {
        "setpoint_1": action_normalizer,
        "setpoint_2": action_normalizer,
        "setpoint_3": action_normalizer,
        "setpoint_4": action_normalizer,
        "setpoint_5": action_normalizer,
        "setpoint_6": action_normalizer,
        "supervisor_run_command": run_command_normalizer,
    }

    return environment.ActionConfig(action_normalizer_inits)

  @parameterized.named_parameters(
      (
          "2_discrete_2_continuous",
          [
              ("boiler_1", "setpoint_1"),
              ("boiler_1", "supervisor_run_command"),
              ("air_handler_5", "setpoint_6"),
              ("air_handler_5", "supervisor_run_command"),
          ],
          2,
          2,
      ),
      (
          "1_discrete_1_continuous",
          [("boiler_1", "setpoint_1"), ("boiler_1", "supervisor_run_command")],
          1,
          1,
      ),
  )
  def test_init_device_action_tuples(
      self, device_action_tuples, num_discrete_actions, num_continuous_actions
  ):
    building = SimpleBuildingHybridAction()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    # Of the 6 possible setpoints, limit only to 2 using device_action_tuples.

    env = hybrid_action_environment.HybridActionEnvironment(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        device_action_tuples=device_action_tuples,
    )
    env.reset()

    expected = {
        "discrete_action": array_spec.BoundedArraySpec(
            (num_discrete_actions,),
            np.int32,
            minimum=0,
            maximum=1,
            name="discrete_action",
        ),
        "continuous_action": array_spec.BoundedArraySpec(
            (num_continuous_actions,),
            np.float32,
            minimum=-1,
            maximum=1,
            name="discrete_action",
        ),
    }

    self.assertEqual(env.action_spec(), expected)

  def test_validate_environment(self):
    class TerminatingEnv(hybrid_action_environment.HybridActionEnvironment):

      def __init__(
          self,
          building: base_building.BaseBuilding,
          reward_function: base_reward_function.BaseRewardFunction,
          obs_normalizer,
          action_config,
          discount_factor: float = 1,
          device_action_tuples=None,
      ):
        super().__init__(
            building,
            reward_function,
            obs_normalizer,
            action_config,
            discount_factor,
            device_action_tuples=device_action_tuples,
        )
        self.counter = 0

      def _step(self, action) -> ts.TimeStep:
        self.counter += 1
        time_step = super()._step(action)
        if self.counter < 100:
          return time_step
        return ts.termination(env._get_observation(), reward=0.0)

    building = SimpleBuildingHybridAction()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    device_action_tuples = [
        ("boiler_1", "setpoint_1"),
        ("boiler_1", "supervisor_run_command"),
        ("air_handler_5", "setpoint_6"),
        ("air_handler_5", "supervisor_run_command"),
    ]
    env = TerminatingEnv(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        1.0,
        device_action_tuples,
    )

    utils.validate_py_environment(env, episodes=5)


if __name__ == "__main__":
  absltest.main()
