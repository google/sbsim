"""Tests for simulator_building.

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
from smart_buildings.smart_control.simulator import simulator_building as sb_py
from smart_buildings.smart_control.simulator import simulator_building_test_lib


class SimulatorBuildingTest(
    simulator_building_test_lib.SimulatorBuildingTestBase
):

  def get_sim_building(
      self, initial_rejection_count: int = 0
  ) -> sb_py.SimulatorBuilding:
    simulator = self._create_small_simulator()
    return sb_py.SimulatorBuilding(simulator, self.occupancy)


if __name__ == '__main__':
  absltest.main()
