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

from absl.testing import absltest
from absl.testing import parameterized
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import rejection_simulator_building as rj_sb_py
from smart_control.simulator import simulator_building as sb_py
from smart_control.simulator import simulator_building_test_lib


class RejectionSimulatorBuildingTest(
    simulator_building_test_lib.SimulatorBuildingTestBase
):

  def get_sim_building(
      self, initial_rejection_count: int = 0
  ) -> rj_sb_py.RejectionSimulatorBuilding:
    simulator = self._create_small_simulator()
    simulator_building = sb_py.SimulatorBuilding(simulator, self.occupancy)
    return rj_sb_py.RejectionSimulatorBuilding(
        simulator_building, initial_rejection_count
    )

  @parameterized.parameters((0), (2), (5))
  def test_request_action_responses_multiple_success_with_rejection(
      self, initial_rejection_count
  ):
    simulator_building = self.get_sim_building(initial_rejection_count)

    action_request = smart_control_building_pb2.ActionRequest()

    single_field_request_1 = smart_control_building_pb2.SingleActionRequest(
        device_id="boiler_id",
        setpoint_name="supply_water_setpoint",
        continuous_value=300,
    )
    action_request.single_action_requests.append(single_field_request_1)

    single_field_request_2 = smart_control_building_pb2.SingleActionRequest(
        device_id="air_handler_id",
        setpoint_name="supply_air_cooling_temperature_setpoint",
        continuous_value=301,
    )
    action_request.single_action_requests.append(single_field_request_2)

    for _ in range(initial_rejection_count):
      with self.assertRaises(RuntimeError):
        _ = simulator_building.request_action(action_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(
        action_response.single_action_responses[0].request,
        single_field_request_1,
    )

    self.assertEqual(
        action_response.single_action_responses[1].request,
        single_field_request_2,
    )

  @parameterized.parameters((0), (2), (5))
  def test_request_action_request_multiple_success_with_rejection(
      self, initial_rejection_count
  ):
    simulator_building = self.get_sim_building(initial_rejection_count)

    action_request = smart_control_building_pb2.ActionRequest()

    single_field_request_1 = smart_control_building_pb2.SingleActionRequest(
        device_id="boiler_id",
        setpoint_name="supply_water_setpoint",
        continuous_value=300,
    )
    action_request.single_action_requests.append(single_field_request_1)

    single_field_request_2 = smart_control_building_pb2.SingleActionRequest(
        device_id="air_handler_id",
        setpoint_name="supply_air_cooling_temperature_setpoint",
        continuous_value=301,
    )
    action_request.single_action_requests.append(single_field_request_2)

    for _ in range(initial_rejection_count):
      with self.assertRaises(RuntimeError):
        _ = simulator_building.request_action(action_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(action_response.request, action_request)

  @parameterized.parameters(
      "devices",
      "zones",
      "current_timestamp",
      "num_occupants",
      "time_step_sec",
  )
  def test_pass_through_properties(self, property_name: str):
    simulator = self._create_small_simulator()
    simulator_building = sb_py.SimulatorBuilding(simulator, self.occupancy)
    base_building = simulator_building
    rejection_simulator = rj_sb_py.RejectionSimulatorBuilding(base_building)
    actual = getattr(rejection_simulator, property_name)
    expected = getattr(base_building, property_name)
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
