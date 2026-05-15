import unittest
from unittest import mock
import pandas as pd
from smart_buildings.smart_control.utils import plot_utils


class PlotUtilsTest(unittest.TestCase):

  def test_update_metrics_recirculation_input(self):
    """Tests how recirculation_input is formed in update_metrics."""
    metrics = plot_utils.init_metrics()
    current_timestamp = pd.Timestamp('2024-01-01 12:00:00')
    current_ambient_temp = 290
    supply_air_temp = 295
    recirculation_temp = 298

    # Mock HVAC and AirHandler components needed for update_metrics
    mock_hvac = mock.Mock()
    mock_hvac.hot_water_system.compute_thermal_energy_rate.return_value = 100
    mock_hvac.hot_water_system.compute_pump_power.return_value = 1
    mock_hvac.air_handler.compute_intake_fan_energy_rate.return_value = 10
    mock_hvac.air_handler.compute_exhaust_fan_energy_rate.return_value = 5
    mock_compute_thermal = mock.Mock(return_value=200)
    mock_hvac.air_handler.compute_thermal_energy_rate = mock_compute_thermal

    # Scenario 1: hvac.air_handler does NOT have 'ahus'
    # Ensure hasattr returns False for 'ahus'
    del mock_hvac.air_handler.ahus

    plot_utils.update_metrics(
        metrics,
        current_timestamp,
        current_ambient_temp,
        supply_air_temp,
        mock_hvac,
        recirculation_temp,
    )
    # In this case, recirculation_input should be just recirculation_temp.
    mock_compute_thermal.assert_called_once_with(
        recirculation_temp, current_ambient_temp
    )
    self.assertEqual(metrics['air_handler_thermal_energy_rates'][-1], 200)
    mock_compute_thermal.reset_mock()

    # Scenario 2: hvac.air_handler HAS 'ahus'
    mock_ahu1 = mock.Mock()
    mock_ahu1.device_id.return_value = 'ahu_1'
    mock_ahu2 = mock.Mock()
    mock_ahu2.device_id.return_value = 'ahu_2'
    mock_hvac.air_handler.ahus = [mock_ahu1, mock_ahu2]

    plot_utils.update_metrics(
        metrics,
        current_timestamp,
        current_ambient_temp,
        supply_air_temp,
        mock_hvac,
        recirculation_temp,
    )
    # In this case, recirculation_input should be a dict mapping device_ids
    # to recirculation_temp.
    expected_recirculation_input = {
        'ahu_1': recirculation_temp,
        'ahu_2': recirculation_temp,
    }
    mock_compute_thermal.assert_called_once_with(
        expected_recirculation_input, current_ambient_temp
    )
    self.assertEqual(metrics['air_handler_thermal_energy_rates'][-1], 200)


if __name__ == '__main__':
  unittest.main()
