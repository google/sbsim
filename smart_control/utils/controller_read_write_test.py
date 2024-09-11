"""Tests for controller_reader and controller_writer.

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

import operator
import os

from absl.testing import absltest
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import controller_reader
from smart_control.utils import controller_writer
from smart_control.utils import conversion_utils


class ControllerReadWriteTest(absltest.TestCase):

  def test_read_write_action_response(self):
    action_responses = [
        (
            '2021-05-25 19:01:01',
            self._get_test_action_response(
                pd.Timestamp('2021-05-25 19:00+0'),
                'ABC',
                'water_valve',
                100.0,
                pd.Timestamp('2021-05-25 19:01+0'),
                smart_control_building_pb2.SingleActionResponse.ACCEPTED,
            ),
        ),
        (
            '2021-05-25 19:08:01',
            self._get_test_action_response(
                pd.Timestamp('2021-05-25 19:05+0'),
                'ABC',
                'airflow',
                25.0,
                pd.Timestamp('2021-05-25 19:07+0'),
                smart_control_building_pb2.SingleActionResponse.ACCEPTED,
            ),
        ),
        (
            '2021-05-25 20:01:01',
            self._get_test_action_response(
                pd.Timestamp('2021-05-25 20:00+0'),
                'XYZ',
                'water_valve',
                100.0,
                pd.Timestamp('2021-05-25 20:01+0'),
                smart_control_building_pb2.SingleActionResponse.ACCEPTED,
            ),
        ),
        (
            '2021-05-25 20:08:01',
            self._get_test_action_response(
                pd.Timestamp('2021-05-25 20:05+0'),
                'XYX',
                'airflow',
                25.0,
                pd.Timestamp('2021-05-25 20:07+0'),
                smart_control_building_pb2.SingleActionResponse.ACCEPTED,
            ),
        ),
    ]

    working_dir = self.create_tempdir()
    writer = controller_writer.ProtoWriter(working_dir)

    for action_response in action_responses:
      writer.write_action_response(
          action_response[1], pd.Timestamp(action_response[0])
      )

    reader = controller_reader.ProtoReader(working_dir)

    action_responses_read = reader.read_action_responses(
        pd.Timestamp('2021-05-25 20:00:00'), pd.Timestamp('2021-05-25 22:00:00')
    )

    action_responses_written = [action_responses[2][1], action_responses[3][1]]
    self.assertListEqual(action_responses_read, action_responses_written)

  def test_get_episode_data(self):
    working_dir = self.create_tempdir()

    expected = pd.DataFrame(
        {
            'execution_time': [
                pd.Timestamp('2022-01-26 14:05:58+00:00'),
                pd.Timestamp('2022-01-26 14:08:39+00:00'),
            ],
            'episode_start_time': [pd.Timestamp('2021-05-25 18:00:00')] * 2,
            'episode_end_time': [pd.Timestamp('2021-05-25 21:00:00')] * 2,
            'duration': [10800.0] * 2,
            'number_updates': [4] * 2,
            'label': ['bc_eval', 'sac_eval'],
        },
        index=['bc_eval_220126_140558', 'sac_eval_220126_140839'],
    )

    observation_responses = [
        (
            '2021-05-25 18:01:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 17:59+0'),
                'ABC',
                'airflow_sensor',
                pd.Timestamp('2021-05-25 18:00+0'),
                True,
                0.9,
            ),
        ),
        (
            '2021-05-25 19:02:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 19:59+0'),
                'XYZ',
                'temp_sensor',
                pd.Timestamp('2021-05-25 20:00+0'),
                True,
                293.0,
            ),
        ),
        (
            '2021-05-25 20:10:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 20:59+0'),
                'ABC',
                'airflow_sensor',
                pd.Timestamp('2021-05-25 21:00+0'),
                True,
                0.9,
            ),
        ),
        (
            '2021-05-25 21:20:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 21:59+0'),
                'XYZ',
                'temp_sensor',
                pd.Timestamp('2021-05-25 22:00+0'),
                False,
                0.0,
            ),
        ),
    ]

    p0 = os.path.join(working_dir, 'bc_eval_220126_140558')
    os.mkdir(p0)
    writer0 = controller_writer.ProtoWriter(p0)
    for observation_response in observation_responses:
      writer0.write_observation_response(
          observation_response[1], pd.Timestamp(observation_response[0])
      )

    p1 = os.path.join(working_dir, 'sac_eval_220126_140839')
    os.mkdir(p1)
    writer1 = controller_writer.ProtoWriter(p1)
    for observation_response in observation_responses:
      writer1.write_observation_response(
          observation_response[1], pd.Timestamp(observation_response[0])
      )

    episode_infos = controller_reader.get_episode_data(working_dir)

    pd.testing.assert_frame_equal(expected, episode_infos)

  def test_read_write_observation_response(self):
    observation_responses = [
        (
            '2021-05-25 20:01:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 19:59+0'),
                'ABC',
                'airflow_sensor',
                pd.Timestamp('2021-05-25 20:00+0'),
                True,
                0.9,
            ),
        ),
        (
            '2021-05-25 20:02:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 19:59+0'),
                'XYZ',
                'temp_sensor',
                pd.Timestamp('2021-05-25 20:00+0'),
                True,
                293.0,
            ),
        ),
        (
            '2021-05-25 20:10:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 19:59+0'),
                'ABC',
                'airflow_sensor',
                pd.Timestamp('2021-05-25 20:00+0'),
                True,
                0.9,
            ),
        ),
        (
            '2021-05-25 20:20:00',
            self._get_test_observation_response(
                pd.Timestamp('2021-05-25 19:59+0'),
                'XYZ',
                'temp_sensor',
                pd.Timestamp('2021-05-25 20:00+0'),
                False,
                0.0,
            ),
        ),
    ]

    working_dir = self.create_tempdir()
    writer = controller_writer.ProtoWriter(working_dir)

    for observation_response in observation_responses:
      writer.write_observation_response(
          observation_response[1], pd.Timestamp(observation_response[0])
      )

    reader = controller_reader.ProtoReader(working_dir)

    observation_responses_read = reader.read_observation_responses(
        pd.Timestamp('2021-05-25 18:00:00'), pd.Timestamp('2021-05-25 20:00:00')
    )
    self.assertListEqual(
        observation_responses_read,
        list(map(operator.itemgetter(1), observation_responses)),
    )

  def test_read_write_reward_response(self):
    reward_responses = [
        (
            '2021-05-25 19:01:01',
            self._get_test_reward_response(100.0, 80.0, 65.2, 12, 0.35, 99.0),
        ),
        (
            '2021-05-25 19:05:01',
            self._get_test_reward_response(101.0, 81.0, 75.2, 12, 0.36, 99.1),
        ),
        (
            '2021-05-25 19:55:01',
            self._get_test_reward_response(102.0, 82.0, 85.2, 13, 0.37, 99.2),
        ),
        (
            '2021-05-25 20:01:01',
            self._get_test_reward_response(103.0, 83.0, 95.2, 14, 0.38, 99.3),
        ),
        (
            '2021-05-25 20:05:01',
            self._get_test_reward_response(104.0, 84.0, 105.2, 15, 0.39, 99.4),
        ),
        (
            '2021-05-25 20:59:59',
            self._get_test_reward_response(105.0, 85.0, 115.2, 16, 0.40, 99.5),
        ),
    ]
    working_dir = self.create_tempdir()
    writer = controller_writer.ProtoWriter(working_dir)

    for reward_response in reward_responses:
      writer.write_reward_response(
          reward_response[1], pd.Timestamp(reward_response[0])
      )

    reader = controller_reader.ProtoReader(working_dir)

    reward_responses_read = reader.read_reward_responses(
        pd.Timestamp('2021-05-25 18:00:00'), pd.Timestamp('2021-05-25 19:00:00')
    )
    self.assertListEqual(
        reward_responses_read,
        list(
            map(
                operator.itemgetter(1),
                reward_responses[: len(reward_responses_read)],
            )
        ),
    )

  def test_read_write_reward_info(self):
    working_dir = self.create_tempdir()
    reward_infos = [
        (
            '2021-05-25 19:01:01',
            self._get_test_reward_info(
                294,
                2.0,
                120.0,
                800.0,
                500.0,
                250.0,
                '2021-05-25 19:01:01',
                '2021-05-25 19:05:05',
            ),
        ),
        (
            '2021-05-25 20:01:01',
            self._get_test_reward_info(
                292,
                12.0,
                120.0,
                600.0,
                500.0,
                250.0,
                '2021-05-25 20:01:01',
                '2021-05-25 20:05:05',
            ),
        ),
        (
            '2021-05-25 20:59:01',
            self._get_test_reward_info(
                288,
                5.0,
                120.0,
                50.0,
                50.0,
                20.0,
                '2021-05-25 20:59:01',
                '2021-05-25 21:01:05',
            ),
        ),
    ]

    writer = controller_writer.ProtoWriter(working_dir)

    for reward_info in reward_infos:
      writer.write_reward_info(reward_info[1], pd.Timestamp(reward_info[0]))

    reader = controller_reader.ProtoReader(working_dir)
    reward_infos_read = reader.read_reward_infos(
        pd.Timestamp('2021-05-25 20:00:00'), pd.Timestamp('2021-05-25 22:00:00')
    )
    reward_infos_expected = [reward_info[1] for reward_info in reward_infos[1:]]
    self.assertLen(reward_infos_read, 2)
    self.assertListEqual(reward_infos_read, reward_infos_expected)

  def test_read_write_normalization_info(self):
    working_dir = self.create_tempdir()
    input_normalization_info = self._get_normalization_constants()
    writer = controller_writer.ProtoWriter(working_dir)
    writer.write_normalization_info(input_normalization_info)
    reader = controller_reader.ProtoReader(working_dir)
    output_normalization_info = reader.read_normalization_info()

    self.assertDictEqual(input_normalization_info, output_normalization_info)

  def test_read_write_device_infos(self):
    working_dir = self.create_tempdir()
    input_device_infos = self._get_device_infos()
    writer = controller_writer.ProtoWriter(working_dir)
    writer.write_device_infos(input_device_infos)
    # Add a duplicate write to verify only one copy is retained.
    writer.write_device_infos(input_device_infos)
    reader = controller_reader.ProtoReader(working_dir)
    output_device_infos = reader.read_device_infos()
    self.assertListEqual(input_device_infos, output_device_infos)

  def test_read_write_zone_infos(self):
    working_dir = self.create_tempdir()
    input_zone_infos = self._get_zone_infos()
    writer = controller_writer.ProtoWriter(working_dir)
    writer.write_zone_infos(input_zone_infos)
    # Add a duplicate write to verify only one copy is retained.
    writer.write_zone_infos(input_zone_infos)
    reader = controller_reader.ProtoReader(working_dir)
    output_zone_infos = reader.read_zone_infos()
    self.assertListEqual(input_zone_infos, output_zone_infos)

  def _get_test_reward_response(
      self,
      agent_reward_value,
      productivity_reward,
      electricity_energy_cost,
      natural_gas_energy_cost,
      carbon_emitted,
      carbon_cost,
  ):
    return smart_control_reward_pb2.RewardResponse(
        agent_reward_value=agent_reward_value,
        productivity_reward=productivity_reward,
        electricity_energy_cost=electricity_energy_cost,
        natural_gas_energy_cost=natural_gas_energy_cost,
        carbon_emitted=carbon_emitted,
        carbon_cost=carbon_cost,
    )

  def _get_test_reward_info(
      self,
      zone_air_temperature=294.0,
      average_occupancy=5.0,
      blower_electrical_energy_rate=800.0,
      air_conditioning_electrical_energy_rate=4500.0,
      natural_gas_heating_energy_rate=5000.0,
      pump_electrical_energy_rate=250.0,
      start_timestamp='2021-05-03 12:13:00-5',
      end_timestamp='2021-05-03 12:18:00-5',
  ):
    heating_setpoint_temperature = 293.0
    cooling_setpoint_temperature = 297.0
    zone_air_flow_rate_setpoint = 0.013
    zone_air_flow_rate = 0.012
    info = smart_control_reward_pb2.RewardInfo()
    info.agent_id = 'test_agent'
    info.scenario_id = 'test_scenario'
    info.start_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(
            pd.Timestamp(start_timestamp)
        )
    )
    info.end_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(pd.Timestamp(end_timestamp))
    )
    zone_info = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo()
    zone_info.heating_setpoint_temperature = heating_setpoint_temperature
    zone_info.cooling_setpoint_temperature = cooling_setpoint_temperature
    zone_info.zone_air_temperature = zone_air_temperature
    zone_info.average_occupancy = average_occupancy
    zone_info.air_flow_rate_setpoint = zone_air_flow_rate_setpoint
    zone_info.air_flow_rate = zone_air_flow_rate
    info.zone_reward_infos['0,0'].CopyFrom(zone_info)
    info.zone_reward_infos['1,1'].CopyFrom(zone_info)

    air_handler_info = (
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo()
    )
    air_handler_info.blower_electrical_energy_rate = (
        blower_electrical_energy_rate
    )
    air_handler_info.air_conditioning_electrical_energy_rate = (
        air_conditioning_electrical_energy_rate
    )
    info.air_handler_reward_infos['air_handler_0'].CopyFrom(air_handler_info)
    info.air_handler_reward_infos['air_handler_1'].CopyFrom(air_handler_info)

    boiler_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo()
    boiler_info.natural_gas_heating_energy_rate = (
        natural_gas_heating_energy_rate
    )
    boiler_info.pump_electrical_energy_rate = pump_electrical_energy_rate
    info.boiler_reward_infos['boiler_0'].CopyFrom(boiler_info)
    info.boiler_reward_infos['boiler_2'].CopyFrom(boiler_info)
    return info

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

  def _get_normalization_constants(self):
    normalization_constants = {}
    normalization_constants['temperature'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='temperature', sample_mean=310.0, sample_variance=50 * 50
        )
    )

    normalization_constants['supply_water_setpoint'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='supply_water_setpoint',
            sample_mean=310.0,
            sample_variance=50 * 50,
        )
    )

    normalization_constants['air_flowrate'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='air_flowrate', sample_mean=0.5, sample_variance=4.0
        )
    )

    normalization_constants['differential_pressure'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='differential_pressure',
            sample_mean=20000.0,
            sample_variance=100000.0,
        )
    )

    normalization_constants['percentage'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='percentage', sample_mean=0.5, sample_variance=0.25
        )
    )

    normalization_constants['request_count'] = (
        smart_control_normalization_pb2.ContinuousVariableInfo(
            id='request_count', sample_mean=9, sample_variance=25.0
        )
    )
    return normalization_constants

  def _get_device_infos(self):
    d0 = smart_control_building_pb2.DeviceInfo(
        device_id='device_00',
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
        device_id='device_01',
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

  def _get_zone_infos(self):
    z0 = smart_control_building_pb2.ZoneInfo(
        zone_id='zone00',
        building_id='US-BLDG-0000',
        zone_description='microkitchen',
        area=900.0,
        zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
        floor=2,
    )
    z1 = smart_control_building_pb2.ZoneInfo(
        zone_id='zone01',
        building_id='US-BLDG-0000',
        zone_description='work area 01',
        area=500.0,
        zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
        floor=1,
    )
    return [z0, z1]


if __name__ == '__main__':
  absltest.main()
