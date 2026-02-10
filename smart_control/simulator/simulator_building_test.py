"""Tests for simulator_building."""

from absl.testing import absltest

from smart_control.simulator import simulator_building as sb_py
from smart_control.simulator import simulator_building_test_lib


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
