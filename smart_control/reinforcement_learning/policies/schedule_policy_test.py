"""Tests for schedule_policy."""

import os

from absl.testing import absltest
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts

from smart_control.environment import environment as env_lib
from smart_control.reinforcement_learning.policies import schedule_policy
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
from smart_control.utils import conversion_utils
from smart_control.utils import regression_building_utils


class CreateBaselineSchedulePolicyTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.env = create_and_setup_environment(
        os.path.join(CONFIG_PATH, 'sim_config_1_day.gin'), metrics_path=None
    )
    cls.tf_env = tf_py_environment.TFPyEnvironment(cls.env)
    cls.policy = schedule_policy.create_baseline_schedule_policy(cls.tf_env)
    cls.first = cls.tf_env.reset()

  def _native_setpoints(self, time_step):
    """Maps the policy's action array to native values per env action."""
    action = self.policy.action(time_step).action.numpy()[0]
    names = list(self.env.action_normalizers)
    self.assertLen(action, len(names))
    return {
        name: self.env.action_normalizers[name].setpoint_value(value)
        for name, value in zip(names, action)
    }

  def _setpoint(self, setpoints, suffix):
    matches = [v for name, v in setpoints.items() if name.endswith(suffix)]
    self.assertLen(matches, 1)
    return matches[0]

  def test_action_sequence_follows_environment_action_order(self):
    names = list(self.env.action_normalizers)
    self.assertEqual(
        [setpoint for _, setpoint in self.policy.action_sequence],
        [
            next(
                s
                for s in (
                    'supply_water_setpoint',
                    'supply_air_heating_temperature_setpoint',
                )
                if name.endswith(s)
            )
            for name in names
        ],
    )

  def test_night_schedule_reaches_the_right_setpoints(self):
    # The episode starts at local midnight on a weekday: night schedule.
    setpoints = self._native_setpoints(self.first)
    self.assertAlmostEqual(
        self._setpoint(setpoints, 'supply_water_setpoint'), 315.0, places=3
    )
    self.assertAlmostEqual(
        self._setpoint(setpoints, 'supply_air_heating_temperature_setpoint'),
        285.0,
        places=3,
    )

  def test_day_schedule_reaches_the_right_setpoints(self):
    # Same observation with the hour-of-day features of a weekday morning.
    morning = self.env.current_simulation_timestamp + pd.Timedelta(
        8, unit='hour'
    )
    rad = conversion_utils.get_radian_time(
        morning, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
    )
    features = regression_building_utils.expand_time_features(
        1, rad, env_lib.HOD_LABEL
    )
    observation = self.first.observation.numpy().copy()
    for (label, index), value in features.items():
      field = self.env.field_names.index(f'{label}_{index}')
      observation[0, field] = value
    time_step = ts.restart(tf.constant(observation, dtype=tf.float32), 1)
    setpoints = self._native_setpoints(time_step)
    self.assertAlmostEqual(
        self._setpoint(setpoints, 'supply_water_setpoint'), 350.0, places=3
    )
    self.assertAlmostEqual(
        self._setpoint(setpoints, 'supply_air_heating_temperature_setpoint'),
        292.0,
        places=3,
    )
    self.assertTrue(np.all(np.isfinite(list(setpoints.values()))))


if __name__ == '__main__':
  absltest.main()
