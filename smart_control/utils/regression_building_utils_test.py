"""Tests for regression_building_utils.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from absl.testing import absltest
import pandas as pd
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator.setpoint_schedule import SetpointSchedule
from smart_control.utils import conversion_utils
from smart_control.utils import regression_building_utils


class RegressionBuildingUtilsTest(absltest.TestCase):

  def test_get_time_feature_names(self):
    time_feature_names = regression_building_utils.get_time_feature_names(
        n=4, label='dom'
    )
    expected_time_feature_names = [
        ('dom', 'cos_000'),
        ('dom', 'cos_001'),
        ('dom', 'cos_002'),
        ('dom', 'cos_003'),
        ('dom', 'sin_000'),
        ('dom', 'sin_001'),
        ('dom', 'sin_002'),
        ('dom', 'sin_003'),
    ]

    self.assertListEqual(expected_time_feature_names, time_feature_names)

  def test_expand_time_features(self):
    time_features = regression_building_utils.expand_time_features(
        n=4, rad=0.0, label='dom'
    )

    expected_time_features = {
        ('dom', 'cos_000'): 1,
        ('dom', 'cos_001'): 6.123233995736766e-17,
        ('dom', 'cos_002'): -1,
        ('dom', 'cos_003'): -1.8369701987210297e-16,
        ('dom', 'sin_000'): 0,
        ('dom', 'sin_001'): 1,
        ('dom', 'sin_002'): 1.2246467991473532e-16,
        ('dom', 'sin_003'): -1,
    }

    self.assertDictEqual(expected_time_features, time_features)

  def test_get_action_tuples(self):
    request_timestamp = pd.Timestamp('2021-01-12 00:01')
    device_id = 'd0'
    setpoint_name = 's0'
    value = 101.001
    response_timestamp = pd.Timestamp('2021-01-12 00:05')
    response_type = regression_building_utils._ActionResponseType.ACCEPTED
    action_response = self._get_test_action_response(
        request_timestamp,
        device_id,
        setpoint_name,
        value,
        response_timestamp,
        response_type,
    )
    action_tuples = regression_building_utils.get_action_tuples(action_response)
    self.assertSetEqual({('action', device_id, setpoint_name)}, action_tuples)

  def test_get_matching_indices(self):
    raw_input_sequence = pd.DataFrame(
        {'a': [1, 2, 3, 4, 5], 'b': [2, 2, 2, 2, 2]},
        index=[
            pd.Timestamp('2021-12-30 12:00:00')
            + pd.Timedelta(5 * i, unit='minute')
            for i in range(5)
        ],
    )
    raw_output_sequence = pd.DataFrame(
        {'c': [5, 4, 3, 2, 1], 'd': [1, 1, 1, 1, 1]},
        index=[
            pd.Timestamp('2021-12-30 12:00:00')
            + pd.Timedelta(5 * i, unit='minute')
            for i in range(5)
        ],
    )

    input_indices, output_indices = (
        regression_building_utils.get_matching_indexes(
            raw_input_sequence,
            raw_output_sequence,
            pd.Timedelta(5, unit='minute'),
        )
    )

    expected_input_indices = [
        pd.Timestamp('2021-12-30 12:00:00'),
        pd.Timestamp('2021-12-30 12:05:00'),
        pd.Timestamp('2021-12-30 12:10:00'),
        pd.Timestamp('2021-12-30 12:15:00'),
    ]
    expected_output_indices = [
        pd.Timestamp('2021-12-30 12:05:00'),
        pd.Timestamp('2021-12-30 12:10:00'),
        pd.Timestamp('2021-12-30 12:15:00'),
        pd.Timestamp('2021-12-30 12:20:00'),
    ]
    self.assertListEqual(expected_input_indices, input_indices)
    self.assertListEqual(expected_output_indices, output_indices)

  def test_get_feature_map(self):
    req_ts = pd.Timestamp('2021-01-12 00:00+0')
    res_ts = pd.Timestamp('2021-01-12 00:10+0')
    device_id = 'd0'
    measurement_name = 's0'
    measurement_value = 294.5
    observation_response = self._get_test_observation_response(
        request_timestamp=req_ts,
        device_id=device_id,
        response_timestamp=res_ts,
        measurement_name=measurement_name,
        value=measurement_value,
        observation_valid=True,
    )
    feature_map = regression_building_utils.get_feature_map(
        observation_response
    )
    expected_feature_map = {
        'timestamp': pd.Timestamp('2021-01-12 00:10:00+0'),
        ('hod', 'cos_000'): 0.9990482215818578,
        ('hod', 'sin_000'): 0.043619387365336,
        ('dow', 'cos_000'): 0.6234898018587336,
        ('dow', 'sin_000'): 0.7818314824680298,
        ('d0', 's0'): 294.5,
    }
    self.assertDictEqual(expected_feature_map, feature_map)

  def test_get_observation_sequence(self):
    req_ts = pd.Timestamp('2021-01-12 00:00')
    res_ts = pd.Timestamp('2021-01-12 00:10')
    dt = pd.Timedelta(5, unit='minute')
    device_id = 'd0'
    measurement_name = 's0'
    obs0 = self._get_test_observation_response(
        request_timestamp=req_ts,
        device_id=device_id,
        response_timestamp=res_ts,
        measurement_name=measurement_name,
        value=294.5,
        observation_valid=True,
    )
    obs1 = self._get_test_observation_response(
        request_timestamp=req_ts + dt,
        device_id=device_id,
        response_timestamp=res_ts + dt,
        measurement_name=measurement_name,
        value=301.0,
        observation_valid=True,
    )

    expected_obs_seq = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2021-01-12 00:10:00+0'),
            pd.Timestamp('2021-01-12 00:15:00+0'),
        ],
        ('hod', 'cos_000'): [0.999048, 0.997859],
        ('hod', 'sin_000'): [0.043619, 0.065403],
        ('dow', 'cos_000'): [0.6234898018587336, 0.6234898018587336],
        ('dow', 'sin_000'): [0.7818314824680298, 0.7818314824680298],
        ('d0', 's0'): [294.5, 301.0],
    })
    obs_seq = regression_building_utils.get_observation_sequence(
        [obs0, obs1], {(device_id, measurement_name)}, 'UTC', n_hod=1, n_dow=1
    )
    pd.testing.assert_frame_equal(expected_obs_seq, obs_seq)

  def test_get_feature_tuples(self):
    req_ts = pd.Timestamp('2021-01-12 00:00')
    res_ts = pd.Timestamp('2021-01-12 00:10')
    device_id = 'd0'
    measurement_name = 's0'
    measurement_value = 294.5
    observation_response = self._get_test_observation_response(
        request_timestamp=req_ts,
        device_id=device_id,
        response_timestamp=res_ts,
        measurement_name=measurement_name,
        value=measurement_value,
        observation_valid=True,
    )
    feature_tuples = regression_building_utils.get_feature_tuples(
        observation_response
    )
    self.assertSetEqual({(device_id, measurement_name)}, feature_tuples)

  def test_get_device_action_tuples(self):
    device_infos = self._get_device_infos()
    device_action_tuples = regression_building_utils.get_device_action_tuples(
        device_infos
    )
    expected_device_action_tuples = [
        ('action', 'device00', 'a0'),
        ('action', 'device00', 'a1'),
        ('action', 'device01', 'a0'),
        ('action', 'device01', 'a1'),
    ]
    self.assertCountEqual(expected_device_action_tuples, device_action_tuples)

  def test_get_reward_info_tuples(self):
    reward_info = self._get_test_reward_info()
    reward_info_tuples = regression_building_utils.get_reward_info_tuples(
        reward_info
    )
    expected_reward_info_tuples = {
        ('reward_info', 'a0', 'air_conditioning_electrical_energy_rate'),
        ('reward_info', 'timestamp', 'end'),
        ('reward_info', 'timestamp', 'start'),
        ('reward_info', 'a0', 'blower_electrical_energy_rate'),
        ('reward_info', 'b1', 'pump_electrical_energy_rate'),
        ('reward_info', 'a1', 'air_conditioning_electrical_energy_rate'),
        ('reward_info', 'b1', 'natural_gas_heating_energy_rate'),
        ('reward_info', 'b0', 'pump_electrical_energy_rate'),
        ('reward_info', 'b0', 'natural_gas_heating_energy_rate'),
        ('reward_info', 'a1', 'blower_electrical_energy_rate'),
    }
    self.assertSetEqual(expected_reward_info_tuples, reward_info_tuples)

  def test_get_reward_info_map(self):
    reward_info = self._get_test_reward_info()
    reward_info_map = regression_building_utils.get_reward_info_map(reward_info)
    expected_reward_info_map = {
        ('reward_info', 'timestamp', 'start'): pd.Timestamp(
            '2022-01-12 12:01:00+0'
        ),
        ('reward_info', 'timestamp', 'end'): pd.Timestamp(
            '2022-01-12 12:06:00+0'
        ),
        ('reward_info', 'a1', 'blower_electrical_energy_rate'): 1236.0,
        (
            'reward_info',
            'a1',
            'air_conditioning_electrical_energy_rate',
        ): 1900.199951171875,
        ('reward_info', 'a0', 'blower_electrical_energy_rate'): 200.0,
        (
            'reward_info',
            'a0',
            'air_conditioning_electrical_energy_rate',
        ): 32.099998474121094,
        ('reward_info', 'b1', 'natural_gas_heating_energy_rate'): 199.5,
        ('reward_info', 'b1', 'pump_electrical_energy_rate'): 12.0,
        ('reward_info', 'b0', 'natural_gas_heating_energy_rate'): 1234.5,
        ('reward_info', 'b0', 'pump_electrical_energy_rate'): 52.0,
    }
    self.assertDictEqual(expected_reward_info_map, reward_info_map)

  def _get_test_reward_info(self):
    start_time = pd.Timestamp('2022-01-12 12:01')
    end_time = pd.Timestamp('2022-01-12 12:06')
    reward_info = smart_control_reward_pb2.RewardInfo(
        start_timestamp=conversion_utils.pandas_to_proto_timestamp(start_time),
        end_timestamp=conversion_utils.pandas_to_proto_timestamp(end_time),
    )
    air_handler0 = 'a0'
    air_conditioning_electrical_energy_rate_d0 = 32.10
    blower_electrical_energy_rate_d0 = 200.0
    air_handler1 = 'a1'
    air_conditioning_electrical_energy_rate_d1 = 1900.20
    blower_electrical_energy_rate_d1 = 1236.0
    reward_info.air_handler_reward_infos[air_handler0].CopyFrom(
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=blower_electrical_energy_rate_d0,
            air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate_d0,
        )
    )
    reward_info.air_handler_reward_infos[air_handler1].CopyFrom(
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=blower_electrical_energy_rate_d1,
            air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate_d1,
        )
    )

    boiler0 = 'b0'
    natural_gas_heating_energy_rate_d0 = 1234.50
    pump_electrical_energy_rate_d0 = 52.0
    boiler1 = 'b1'
    natural_gas_heating_energy_rate_d1 = 199.50
    pump_electrical_energy_rate_d1 = 12.0
    reward_info.boiler_reward_infos[boiler0].CopyFrom(
        smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=natural_gas_heating_energy_rate_d0,
            pump_electrical_energy_rate=pump_electrical_energy_rate_d0,
        )
    )
    reward_info.boiler_reward_infos[boiler1].CopyFrom(
        smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=natural_gas_heating_energy_rate_d1,
            pump_electrical_energy_rate=pump_electrical_energy_rate_d1,
        )
    )

    return reward_info

  def test_get_observation_response(self):
    req_ts = pd.Timestamp('2021-01-12 00:00')
    res_ts = pd.Timestamp('2021-01-12 00:10')
    device_id = 'd0'
    measurement_name = 's0'
    measurement_value = 294.5
    expected_observation_response = self._get_test_observation_response(
        request_timestamp=req_ts,
        device_id=device_id,
        response_timestamp=res_ts,
        measurement_name=measurement_name,
        value=measurement_value,
        observation_valid=True,
    )
    observation_response = regression_building_utils.get_observation_response(
        expected_observation_response.request,
        {(device_id, measurement_name): measurement_value},
        current_timestamp=res_ts,
    )
    self.assertEqual(expected_observation_response, observation_response)

  def test_observation_response_to_observation_mapping(self):
    req_ts = pd.Timestamp('2021-01-12 00:00')
    res_ts = pd.Timestamp('2021-01-12 00:10')
    device_id = 'd0'
    measurement_name = 's0'
    measurement_value = 294.5
    observation_response = self._get_test_observation_response(
        request_timestamp=req_ts,
        device_id=device_id,
        response_timestamp=res_ts,
        measurement_name=measurement_name,
        value=measurement_value,
        observation_valid=True,
    )
    expected_mapping = {(device_id, measurement_name): measurement_value}
    mapping = (
        regression_building_utils.observation_response_to_observation_mapping(
            observation_response
        )
    )
    self.assertDictEqual(expected_mapping, mapping)

  def test_create_action_response_valid(self):
    request_timestamp = pd.Timestamp('2021-01-12 00:01')
    device_id = 'd0'
    setpoint_name = 's0'
    value = 101.001
    response_timestamp = pd.Timestamp('2021-01-12 00:05')
    response_type = regression_building_utils._ActionResponseType.ACCEPTED
    expected_action_response = self._get_test_action_response(
        request_timestamp,
        device_id,
        setpoint_name,
        value,
        response_timestamp,
        response_type,
    )
    device_action_tuples = [('action', device_id, setpoint_name)]
    action_response = regression_building_utils.create_action_response(
        expected_action_response.request,
        response_timestamp,
        device_action_tuples,
    )
    self.assertEqual(expected_action_response, action_response)

  def test_create_action_response_not_valid(self):
    request_timestamp = pd.Timestamp('2021-01-12 00:01')
    device_id = 'd0'
    setpoint_name = 's0'
    value = 101.001
    response_timestamp = pd.Timestamp('2021-01-12 00:05')
    response_type = (
        regression_building_utils._ActionResponseType.REJECTED_INVALID_DEVICE
    )
    expected_action_response = self._get_test_action_response(
        request_timestamp,
        device_id,
        setpoint_name,
        value,
        response_timestamp,
        response_type,
    )
    device_action_tuples = [('action', 'd1', 's1')]
    action_response = regression_building_utils.create_action_response(
        expected_action_response.request,
        response_timestamp,
        device_action_tuples,
    )
    self.assertEqual(expected_action_response, action_response)

  def test_split_output_into_observations_and_reward_info_mapping(self):
    expected_observation_mapping = {
        ('d0', 's0'): 0.1,
        ('d0', 's1'): 293.0,
        ('d1', 's0'): 0.658,
    }
    expected_reward_info_mapping = {
        ('reward_info', 'r0', 'v0'): 0.1,
        ('reward_info', 'r1', 'v2'): 100,
    }

    input_mapping = expected_observation_mapping.copy()
    input_mapping.update(expected_reward_info_mapping)
    observation_mapping, reward_info_mapping = (
        regression_building_utils.split_output_into_observations_and_reward_info_mapping(
            input_mapping
        )
    )
    self.assertDictEqual(expected_observation_mapping, observation_mapping)
    self.assertDictEqual(expected_reward_info_mapping, reward_info_mapping)

  def test_action_request_to_action_mapping(self):
    request_timestamp = pd.Timestamp('2021-01-12 00:01')
    device_id = 'd0'
    setpoint_name = 's0'
    value = 101.0
    response_timestamp = pd.Timestamp('2021-01-12 00:05')
    response_type = regression_building_utils._ActionResponseType.ACCEPTED
    action_request = self._get_test_action_response(
        request_timestamp,
        device_id,
        setpoint_name,
        value,
        response_timestamp,
        response_type,
    ).request
    device_action_tuples = [('action', device_id, setpoint_name)]
    expected_mapping = {('action', device_id, setpoint_name): value}
    mapping = regression_building_utils.action_request_to_action_mapping(
        action_request, device_action_tuples
    )
    self.assertDictEqual(expected_mapping, mapping)

  def test_current_device_observations(self):
    current_observations = {
        ('d0', 's0'): 3.0,
        ('d0', 's1'): 0.1,
        ('d1', 's0'): 8.0,
        ('d1', 's1'): 80.0,
    }
    expected_device_observations = {'s0': 8.0, 's1': 80.0}
    device_observations = (
        regression_building_utils.get_current_device_observations(
            current_observations, 'd1'
        )
    )
    self.assertDictEqual(expected_device_observations, device_observations)

  def test_get_boiler_reward_infos(self):
    device0 = 'd0'
    natural_gas_heating_energy_rate_d0 = 1234.50
    pump_electrical_energy_rate_d0 = 52.0
    device1 = 'd1'
    natural_gas_heating_energy_rate_d1 = 199.50
    pump_electrical_energy_rate_d1 = 12.0
    expected_boiler_reward_infos = {
        device0: smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=natural_gas_heating_energy_rate_d0,
            pump_electrical_energy_rate=pump_electrical_energy_rate_d0,
        ),
        device1: smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=natural_gas_heating_energy_rate_d1,
            pump_electrical_energy_rate=pump_electrical_energy_rate_d1,
        ),
    }
    reward_info_devices = {
        device0: {
            'natural_gas_heating_energy_rate': (
                natural_gas_heating_energy_rate_d0
            ),
            'pump_electrical_energy_rate': pump_electrical_energy_rate_d0,
        },
        device1: {
            'natural_gas_heating_energy_rate': (
                natural_gas_heating_energy_rate_d1
            ),
            'pump_electrical_energy_rate': pump_electrical_energy_rate_d1,
        },
    }
    boiler_reward_infos = regression_building_utils.get_boiler_reward_infos(
        reward_info_devices
    )
    self.assertEqual(
        expected_boiler_reward_infos[device0],
        boiler_reward_infos[device0],
    )
    self.assertEqual(
        expected_boiler_reward_infos[device1],
        boiler_reward_infos[device1],
    )

  def test_get_air_handler_reward_infos(self):
    device0 = 'd0'
    air_conditioning_electrical_energy_rate_d0 = 32.10
    blower_electrical_energy_rate_d0 = 200.0
    device1 = 'd1'
    air_conditioning_electrical_energy_rate_d1 = 1900.20
    blower_electrical_energy_rate_d1 = 1236.0
    expected_air_handler_reward_infos = {
        device0: smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=blower_electrical_energy_rate_d0,
            air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate_d0,
        ),
        device1: smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=blower_electrical_energy_rate_d1,
            air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate_d1,
        ),
    }
    reward_info_devices = {
        device0: {
            'blower_electrical_energy_rate': blower_electrical_energy_rate_d0,
            'air_conditioning_electrical_energy_rate': (
                air_conditioning_electrical_energy_rate_d0
            ),
        },
        device1: {
            'blower_electrical_energy_rate': blower_electrical_energy_rate_d1,
            'air_conditioning_electrical_energy_rate': (
                air_conditioning_electrical_energy_rate_d1
            ),
        },
    }
    air_handler_reward_infos = (
        regression_building_utils.get_air_handler_reward_infos(
            reward_info_devices
        )
    )
    self.assertEqual(
        expected_air_handler_reward_infos[device0],
        air_handler_reward_infos[device0],
    )
    self.assertEqual(
        expected_air_handler_reward_infos[device1],
        air_handler_reward_infos[device1],
    )

  def test_get_zone_reward_infos_valid(self):
    setpoint_schedule = SetpointSchedule(
        morning_start_hour=7,
        evening_start_hour=18,
        comfort_temp_window=(291.0, 295.0),
        eco_temp_window=(287.0, 298.0),
    )

    occupancy = self.TestOccupancy()
    current_timestamp = pd.Timestamp('2021-01-14 14:01+0')
    step_interval = pd.Timedelta(5, unit='minute')
    zone_infos, device_infos = self._get_test_zone_device_infos()
    current_observation_mapping = {
        ('d0', 'zone_air_temperature_sensor'): 72.0,
        ('d1', 'zone_air_temperature_sensor'): 68.0,
    }
    zone_reward_infos = regression_building_utils.get_zone_reward_infos(
        current_timestamp=current_timestamp,
        step_interval=step_interval,
        occupancy_function=occupancy,
        setpoint_schedule=setpoint_schedule,
        zone_infos=zone_infos,
        device_infos=device_infos,
        current_observation_mapping=current_observation_mapping,
    )

    expected_zone_reward_infos = {
        'z0': smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
            heating_setpoint_temperature=291.0,
            cooling_setpoint_temperature=295.0,
            zone_air_temperature=295.37222,
            average_occupancy=10.0,
        ),
        'z1': smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
            heating_setpoint_temperature=291.0,
            cooling_setpoint_temperature=295.0,
            zone_air_temperature=293.15,
            average_occupancy=10.0,
        ),
    }
    self.assertAlmostEqual(
        expected_zone_reward_infos['z0'],
        zone_reward_infos['z0'],
        delta=0.001,
    )
    self.assertAlmostEqual(
        expected_zone_reward_infos['z1'],
        zone_reward_infos['z1'],
        delta=0.001,
    )

  def test_get_zone_reward_infos_invalid(self):
    class BadSetpointSchedule(SetpointSchedule):

      def get_temperature_window(
          self, current_timestamp: pd.Timestamp
      ) -> tuple[float, float]:
        """Heating setpoint should always be less than cooling setpoint."""
        return (291.0, 290.0)

    bad_setpoint_schedule = BadSetpointSchedule(
        morning_start_hour=7,
        evening_start_hour=18,
        comfort_temp_window=(291.0, 296.0),
        eco_temp_window=(287.0, 298.0),
    )

    occupancy = self.TestOccupancy()
    current_timestamp = pd.Timestamp('2021-01-14 14:01+0')
    step_interval = pd.Timedelta(5, unit='minute')
    zone_infos, device_infos = self._get_test_zone_device_infos()
    current_observation_mapping = {
        ('d0', 'zone_air_temperature_sensor'): 72.0,
        ('d1', 'zone_air_temperature_sensor'): 68.0,
    }
    with self.assertRaisesRegex(
        ValueError,
        'Bad setpoints: zone_air_heating_temperature_setpoint 291.0 >'
        ' zone_air_cooling_temperature_setpoint 290.0',
    ):
      _ = regression_building_utils.get_zone_reward_infos(
          current_timestamp=current_timestamp,
          step_interval=step_interval,
          occupancy_function=occupancy,
          setpoint_schedule=bad_setpoint_schedule,
          zone_infos=zone_infos,
          device_infos=device_infos,
          current_observation_mapping=current_observation_mapping,
      )

  # Test Utilities #

  def _get_test_action_response(
      self,
      request_timestamp,
      device_id,
      setpoint_name,
      value,
      response_timestamp,
      response_type,
  ):
    request_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(request_timestamp)
    )
    response_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(response_timestamp)
    )
    single_request = smart_control_building_pb2.SingleActionRequest(
        device_id=device_id, setpoint_name=setpoint_name, continuous_value=value
    )
    single_response = smart_control_building_pb2.SingleActionResponse(
        request=single_request, response_type=response_type
    )
    request = smart_control_building_pb2.ActionRequest(
        timestamp=request_ts, single_action_requests=[single_request]
    )
    return smart_control_building_pb2.ActionResponse(
        timestamp=response_ts,
        request=request,
        single_action_responses=[single_response],
    )

  def _get_test_zone_device_infos(self):
    z0 = smart_control_building_pb2.ZoneInfo(
        zone_id='z0',
        building_id='US-SIM-0000',
        zone_description='test zone',
        area=121.0,
        zone_type=smart_control_building_pb2.ZoneInfo.ZoneType.ROOM,
        floor=2,
        devices=['d0'],
    )

    d0 = smart_control_building_pb2.DeviceInfo(
        device_id='d0',
        namespace='test',
        code='code0',
        zone_id='z0',
        device_type=smart_control_building_pb2.DeviceInfo.VAV,
        observable_fields={
            'zone_air_temperature_sensor': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            ),
        },
    )

    z1 = smart_control_building_pb2.ZoneInfo(
        zone_id='z1',
        building_id='US-SIM-0000',
        zone_description='test zone',
        area=999.0,
        zone_type=smart_control_building_pb2.ZoneInfo.ZoneType.ROOM,
        floor=1,
        devices=['d1'],
    )

    d1 = smart_control_building_pb2.DeviceInfo(
        device_id='d1',
        namespace='test',
        code='code0',
        zone_id='z1',
        device_type=smart_control_building_pb2.DeviceInfo.VAV,
        observable_fields={
            'zone_air_temperature_sensor': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            ),
        },
    )

    return [z0, z1], [d0, d1]

  class TestOccupancy(BaseOccupancy):

    def average_zone_occupancy(
        self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
    ):
      return 10.0

  def _get_device_infos(self):
    d0 = smart_control_building_pb2.DeviceInfo(
        device_id='device00',
        namespace='test',
        code='code0',
        zone_id='zone00',
        device_type=smart_control_building_pb2.DeviceInfo.AHU,
        observable_fields={
            'f0': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            ),
            'f1': smart_control_building_pb2.DeviceInfo.ValueType.VALUE_INTEGER,
        },
        action_fields={
            'a0': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CATEGORICAL
            ),
            'a1': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            ),
        },
    )

    d1 = smart_control_building_pb2.DeviceInfo(
        device_id='device01',
        namespace='test',
        code='code1',
        zone_id='zone01',
        device_type=smart_control_building_pb2.DeviceInfo.AHU,
        observable_fields={
            'f0': smart_control_building_pb2.DeviceInfo.ValueType.VALUE_BINARY,
            'f1': smart_control_building_pb2.DeviceInfo.ValueType.VALUE_INTEGER,
        },
        action_fields={
            'a0': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_TYPE_UNDEFINED
            ),
            'a1': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            ),
        },
    )
    return [d0, d1]

  def _get_test_observation_response(
      self,
      request_timestamp,
      device_id,
      measurement_name,
      response_timestamp,
      observation_valid,
      value,
  ):
    request_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(request_timestamp)
    )
    response_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(response_timestamp)
    )
    single_request = smart_control_building_pb2.SingleObservationRequest(
        device_id=device_id, measurement_name=measurement_name
    )
    single_response = smart_control_building_pb2.SingleObservationResponse(
        timestamp=response_ts,
        single_observation_request=single_request,
        observation_valid=observation_valid,
        continuous_value=value,
    )
    request = smart_control_building_pb2.ObservationRequest(
        timestamp=request_ts, single_observation_requests=[single_request]
    )
    return smart_control_building_pb2.ObservationResponse(
        timestamp=response_ts,
        request=request,
        single_observation_responses=[single_response],
    )


if __name__ == '__main__':
  absltest.main()
