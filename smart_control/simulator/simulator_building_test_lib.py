"""Tests for rejection_simulator_building.

Copyright 2023 Google LLC

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

from absl.testing import parameterized
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac as hvac_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import simulator as simulator_py
from smart_control.simulator import simulator_building as sb_py
from smart_control.simulator import step_function_occupancy
from smart_control.simulator import weather_controller as weather_controller_py

_ACTION_RESPONSE_TYPE = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)


class SimulatorBuildingTestBase(parameterized.TestCase):
  """Base class for testing variants of Simulator Building."""

  occupancy = step_function_occupancy.StepFunctionOccupancy(
      pd.Timedelta(9, unit='h'), pd.Timedelta(17, unit='h'), 10, 0.1
  )

  def _create_small_building(self, initial_temp):
    """Returns building with specified initial temperature.

    The building returned will have a matrix size of: 21 x 10, this should be
    left as a comment in tests when relevant. Additionally initial_temp is
    added as a parameter for clarity in the tests.

    Args:
      initial_temp: Initial temperature of all CVs in building.
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (8, 6)
    building_shape = (2, 1)
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=500.0, density=3000.0
    )

    building = building_py.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )
    return building

  def _create_small_hvac(self):
    """Returns hvac matching zones for small test building."""
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    boiler = boiler_py.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    recirculation = 0.3
    heating_air_temp_setpoint = 270
    cooling_air_temp_setpoint = 288
    fan_differential_pressure = 20000.0
    fan_efficiency = 0.8

    air_handler = air_handler_py.AirHandler(
        recirculation,
        heating_air_temp_setpoint,
        cooling_air_temp_setpoint,
        fan_differential_pressure,
        fan_efficiency,
        device_id='air_handler_id',
    )

    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 223, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    zone_coordinates = [(0, 0), (1, 0)]

    hvac = hvac_py.Hvac(
        zone_coordinates, air_handler, boiler, schedule, 0.45, 0.02
    )
    return hvac

  def _create_small_simulator(self):
    """Creats a small simulator for test."""
    # Set up simulation parameters
    weather_controller = weather_controller_py.WeatherController(296.0, 296.0)
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('2012-12-21')

    # Control Volume matrix shape: (21, 10)
    building = self._create_small_building(initial_temp=292.0)

    return simulator_py.Simulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

  def get_sim_building(
      self, initial_rejection_count: int = 0
  ) -> sb_py.SimulatorBuilding:
    raise NotImplementedError()  # pragma: nocover

  def test_devices(self):
    simulator_building = self.get_sim_building()

    devices = simulator_building.devices

    self.assertLen(devices, 4)

  @parameterized.named_parameters(
      ('obs_supply_water_setpoint', 'supply_water_setpoint', 260),
      (
          'obs_supply_water_temperature_sensor',
          'supply_water_temperature_sensor',
          260,
      ),
      ('obs_heating_request_count', 'heating_request_count', 0),
  )
  def test_request_observation_single_success(
      self, measurement_name, expected_value
  ):
    """Tests request observations."""
    simulator_building = self.get_sim_building()

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='boiler_id', measurement_name=measurement_name
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertEqual(observation_response.request, observation_request)
    self.assertEqual(
        observation_response.single_observation_responses[
            0
        ].single_observation_request,
        single_field_request,
    )
    self.assertEqual(
        observation_response.single_observation_responses[0].continuous_value,
        expected_value,
    )
    self.assertTrue(
        observation_response.single_observation_responses[0].observation_valid
    )

  def test_request_observation_multiple_success(self):
    """Tests request multiple observations."""
    simulator_building = self.get_sim_building()

    observation_request = smart_control_building_pb2.ObservationRequest()

    single_field_request_1 = (
        smart_control_building_pb2.SingleObservationRequest(
            device_id='boiler_id', measurement_name='supply_water_setpoint'
        )
    )
    observation_request.single_observation_requests.append(
        single_field_request_1
    )

    single_field_request_2 = (
        smart_control_building_pb2.SingleObservationRequest(
            device_id='boiler_id', measurement_name='heating_request_count'
        )
    )
    observation_request.single_observation_requests.append(
        single_field_request_2
    )

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertEqual(observation_response.request, observation_request)

    self.assertEqual(
        observation_response.single_observation_responses[
            0
        ].single_observation_request,
        single_field_request_1,
    )
    self.assertEqual(
        observation_response.single_observation_responses[0].continuous_value,
        260,
    )

    self.assertEqual(
        observation_response.single_observation_responses[
            1
        ].single_observation_request,
        single_field_request_2,
    )
    self.assertEqual(
        observation_response.single_observation_responses[1].continuous_value, 0
    )

  def test_request_observation_incorrect_device(self):
    """Tests when an observation is requested on a nonexistent device."""
    simulator_building = self.get_sim_building()

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='wrong_device', measurement_name='measurement_name'
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertFalse(
        observation_response.single_observation_responses[0].observation_valid
    )

  def test_request_observation_incorrect_measurement(self):
    """Tests when an observation is requested for a nonexistnt measurement."""
    simulator_building = self.get_sim_building()

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='boiler_id', measurement_name='incorrect_measurement'
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertFalse(
        observation_response.single_observation_responses[0].observation_valid
    )

  @parameterized.named_parameters(
      ('act_supply_water_setpoint', 'supply_water_setpoint', 301),
  )
  def test_request_action_single_success(self, setpoint_name, set_value):
    """Tests request single action with success."""
    simulator_building = self.get_sim_building()

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='boiler_id',
        setpoint_name=setpoint_name,
        continuous_value=set_value,
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(action_response.request, action_request)
    self.assertEqual(
        action_response.single_action_responses[0].request, single_field_request
    )

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.ACCEPTED,
    )

  def test_request_action_incorrect_device(self):
    """Tests when an action is sent to a nonexistent device."""
    simulator_building = self.get_sim_building()

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='wrong_device', setpoint_name='setpoint_name'
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.REJECTED_INVALID_DEVICE,
    )

  def test_request_action_incorrect_setpoint(self):
    """Tests when an action is sent to a nonexistent setpoint."""
    simulator_building = self.get_sim_building()

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='boiler_id', setpoint_name='incorrect_setpoint'
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.REJECTED_NOT_ENABLED_OR_AVAILABLE,
    )
