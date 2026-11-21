"""Tests for LLM response output schema models.

These tests ensure the output schema models can be initialized. However, the
promptmaker actually uses them to generate formatting instructions. Tests for
that functionality are defined in the "formatting_instructions_test.py" file.
"""

from typing import get_args

from absl.testing import absltest
import pydantic
from smart_buildings.smart_control.llm.schema import conftest
from smart_buildings.smart_control.llm.schema import output_schema

DeviceSetpoint = output_schema.DeviceSetpoint
SetpointsAction = output_schema.SetpointsAction

EXAMPLE_TIMESTAMP = conftest.EXAMPLE_TIMESTAMP
EXAMPLE_JUSTIFICATION = conftest.EXAMPLE_JUSTIFICATION


#
# ACTIONS (CONTINUOUS)
#


class ActionValidationsTest(absltest.TestCase):
  """Tests for Pydantic model validations, for continuous actions.

  This ensures the model will raise errors if required fields are missing, or if
  the data is otherwise not in the expected format.
  """

  def setUp(self):
    super().setUp()
    self.creation_function = conftest.create_action_response

  def test_valid_setpoints(self):
    response_text = self.creation_function()
    action = SetpointsAction.model_validate_json(response_text)
    self.assertIsInstance(action, SetpointsAction)

  def test_empty_setpoints_raises(self):
    response_text = self.creation_function(empty_setpoints=True)
    with self.assertRaisesRegex(
        pydantic.ValidationError, "setpoints list cannot be empty"
    ):
      SetpointsAction.model_validate_json(response_text)

  def test_missing_setpoint_ok_beware(self):
    # The schema doesn't know about which of the environment's setpoints are
    # required. Those validations should happen at the environment level.
    response_text = self.creation_function(missing_setpoint=True)
    action = SetpointsAction.model_validate_json(response_text)
    self.assertIsInstance(action, SetpointsAction)

  def test_missing_field_raises(self):
    response_text = self.creation_function(missing_field=True)
    with self.assertRaisesRegex(pydantic.ValidationError, "Field required"):
      SetpointsAction.model_validate_json(response_text)


class ActionTest(absltest.TestCase):
  """Tests for the basic action model that uses default validity intervals."""

  def setUp(self):
    super().setUp()
    self.n_setpoints_expected = 3
    self.expected_setpoint_names = [
        "supply_air_heating_temperature_setpoint",
        "supply_air_heating_temperature_setpoint",
        "supply_water_setpoint",
    ]
    self.action = conftest.create_action()

  def test_validity_interval_options(self):
    self.assertCountEqual(
        get_args(self.action.__class__.__annotations__["validity_interval"]),
        output_schema.DEFAULT_VALIDITY_INTERVALS,
    )

  def test_initialization(self):
    self.assertIsInstance(self.action, SetpointsAction)

  def test_attributes(self):
    with self.subTest("timestamp"):
      self.assertEqual(self.action.timestamp, EXAMPLE_TIMESTAMP)

    with self.subTest("justification"):
      self.assertEqual(self.action.justification, EXAMPLE_JUSTIFICATION)

    with self.subTest("validity_interval"):
      self.assertEqual(self.action.validity_interval, 60)

    with self.subTest("setpoints"):
      self.assertLen(self.action.setpoints, self.n_setpoints_expected)

      names = [setpoint.setpoint_name for setpoint in self.action.setpoints]
      self.assertEqual(names, self.expected_setpoint_names)

      for i, setpoint in enumerate(self.action.setpoints):
        with self.subTest(f"setpoint at index {i}"):
          self.assertIsInstance(setpoint, DeviceSetpoint)

  # TESTS FOR FIND_SETPOINT METHOD:

  def test_find_setpoint_invalid_device_id(self):
    setpoint = self.action.find_setpoint(
        device_id="oops", setpoint_name="supply_water_setpoint"
    )
    self.assertIsNone(setpoint)

  def test_find_setpoint_invalid_setpoint_name(self):
    setpoint = self.action.find_setpoint(
        device_id="boiler_0", setpoint_name="oops"
    )
    self.assertIsNone(setpoint)

  def test_find_setpoint(self):
    setpoint = self.action.find_setpoint(
        device_id="boiler_1", setpoint_name="supply_water_setpoint"
    )
    self.assertIsInstance(setpoint, DeviceSetpoint)

    with self.subTest("attributes"):
      self.assertEqual(setpoint.device_id, "boiler_1")
      self.assertEqual(setpoint.setpoint_name, "supply_water_setpoint")
      self.assertEqual(setpoint.setpoint_value, 310.0)


class ActionWithCustomValidityIntervalsTest(ActionTest):
  """Tests for the action model that uses custom validity intervals."""

  def setUp(self):
    super().setUp()
    self.custom_intervals = [15, 30, 45, 60]
    self.action = conftest.create_action_with_custom_intervals(
        validity_intervals=self.custom_intervals,
        selected_interval=60,
    )

  def test_validity_interval_options(self):
    self.assertCountEqual(
        get_args(self.action.__class__.__annotations__["validity_interval"]),
        self.custom_intervals,
    )


#
# ACTIONS (HYBRID)
#


class HybridActionValidationsTest(ActionValidationsTest):
  """Tests for Pydantic model validations, for hybrid actions.

  This ensures the model will raise errors if required fields are missing, or if
  the data is otherwise not in the expected format.
  """

  def setUp(self):
    super().setUp()
    self.creation_function = conftest.create_hybrid_action_response


class HybridActionTest(ActionTest):
  """Tests for the hybrid action model that uses default validity intervals."""

  def setUp(self):
    super().setUp()
    self.n_setpoints_expected = 6
    self.expected_setpoint_names = [
        "supervisor_run_command",
        "supervisor_run_command",
        "supervisor_run_command",
        "supply_air_heating_temperature_setpoint",
        "supply_air_heating_temperature_setpoint",
        "supply_water_setpoint",
    ]
    self.action = conftest.create_hybrid_action()


class HybridActionWithCustomValidityIntervalsTest(HybridActionTest):
  """Tests for the hybrid action model that uses custom validity intervals."""

  def setUp(self):
    super().setUp()
    self.custom_intervals = [15, 30, 45, 60]
    self.action = conftest.create_hybrid_action_with_custom_intervals(
        validity_intervals=self.custom_intervals,
        selected_interval=60,
    )

  def test_validity_interval_options(self):
    self.assertCountEqual(
        get_args(self.action.__class__.__annotations__["validity_interval"]),
        self.custom_intervals,
    )


if __name__ == "__main__":
  absltest.main()
