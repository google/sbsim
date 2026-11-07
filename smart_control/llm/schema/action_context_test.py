from typing import get_args
from unittest import mock

from absl.testing import absltest
import pandas as pd
import pydantic
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.schema import conftest as schema_conftest
from smart_buildings.smart_control.llm.schema import output_schema_test


class ActionContextTest(output_schema_test.ActionTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_environment(layout=env_conftest.DEMO_LAYOUT)
    self.action_ctx = schema_conftest.create_action_context(env=self.env)

  def test_initialization(self):
    self.assertIsInstance(self.action_ctx, action_context.ActionContext)

  def test_env(self):
    self.assertIsInstance(self.action_ctx.env, environment.Environment)

  def test_clip(self):
    self.assertTrue(self.action_ctx.clip)

  def test_guardrails_exceeded(self):
    self.assertEmpty(self.action_ctx.guardrails_exceeded)

  def test_sorted_setpoints(self):
    names_from_setpoints = [
        (sp.device_id, sp.setpoint_name)
        for sp in self.action_ctx.sorted_setpoints
    ]
    names_from_env = [
        self.env.id_map.inv[action_name]
        for action_name in self.env.action_names
    ]
    self.assertEqual(names_from_setpoints, names_from_env)

  def test_get_action_values(self):
    self.assertEqual(self.action_ctx.get_action_values(), [-1.0, -1.0, -1.0])

  def test_get_action_values_normalizer_not_found_raises(self):
    with mock.patch.dict(self.env.action_normalizers, {}, clear=True):
      with self.assertRaisesRegex(
          ValueError,
          "No normalizer found for setpoint:"
          " 'air_handler_1_supply_air_heating_temperature_setpoint'.",
      ):
        self.action_ctx.get_action_values()

  def test_get_action_values_device_id_not_found_raises(self):
    self.action_ctx.setpoints[0].device_id = "OOPS"
    with self.assertRaisesRegex(
        KeyError, "\\('OOPS', 'supply_air_heating_temperature_setpoint'\\)"
    ):
      self.action_ctx.get_action_values()

  def test_get_action_values_setpoint_name_not_found_raises(self):
    self.action_ctx.setpoints[0].setpoint_name = "OOPS"
    with self.assertRaisesRegex(KeyError, "\\('air_handler_1', 'OOPS'\\)"):
      self.action_ctx.get_action_values()

  def test_setpoints_df(self):
    df = self.action_ctx.setpoints_df
    expected_df = pd.DataFrame([
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": (
                "air_handler_1_supply_air_heating_temperature_setpoint"
            ),
            "device_id": "air_handler_1",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "setpoint_value": 285.0,
            "setpoint_justification": "To cool the air.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": "boiler_1_supply_water_setpoint",
            "device_id": "boiler_1",
            "setpoint_name": "supply_water_setpoint",
            "setpoint_value": 310.0,
            "setpoint_justification": "To heat the water.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": (
                "air_handler_2_supply_air_heating_temperature_setpoint"
            ),
            "device_id": "air_handler_2",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "setpoint_value": 285.0,
            "setpoint_justification": "To cool the air.",
        },
    ])
    pd.testing.assert_frame_equal(df, expected_df)

  def test_flattened_setpoints_record(self):
    record = self.action_ctx.flattened_setpoints_record
    expected_record = {
        "timestamp": "2025-01-01 12:00:00",
        "validity_interval": 60,
        "justification": "These are my overall goals.",
        "air_handler_1_supply_air_heating_temperature_setpoint": 285.0,
        "air_handler_1_supply_air_heating_temperature_setpoint_justification": (
            "To cool the air."
        ),
        "boiler_1_supply_water_setpoint": 310.0,
        "boiler_1_supply_water_setpoint_justification": "To heat the water.",
        "air_handler_2_supply_air_heating_temperature_setpoint": 285.0,
        "air_handler_2_supply_air_heating_temperature_setpoint_justification": (
            "To cool the air."
        ),
    }
    self.assertDictEqual(record, expected_record)


class HybridActionContextTest(output_schema_test.HybridActionTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.action_ctx = schema_conftest.create_hybrid_action_context(env=self.env)

  def test_initialization(self):
    self.assertIsInstance(self.action_ctx, action_context.HybridActionContext)

  def test_env(self):
    self.assertIsInstance(
        self.action_ctx.env,
        hybrid_action_environment.HybridActionEnvironment,
    )

  def test_clip(self):
    self.assertTrue(self.action_ctx.clip)

  def test_guardrails_exceeded(self):
    self.assertEmpty(self.action_ctx.guardrails_exceeded)

  def test_get_action_values(self):
    self.assertEqual(
        self.action_ctx.get_action_values(), [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    )

  def test_get_action_values_normalizer_not_found_raises(self):
    with mock.patch.dict(self.env.action_normalizers, {}, clear=True):
      with self.assertRaisesRegex(
          ValueError,
          "No normalizer found for setpoint:"
          " 'air_handler_1_supply_air_heating_temperature_setpoint'.",
      ):
        self.action_ctx.get_action_values()

  def test_get_action_values_device_id_not_found_raises(self):
    self.action_ctx.setpoints[0].device_id = "OOPS"
    with self.assertRaisesRegex(
        KeyError, "\\('OOPS', 'supervisor_run_command'\\)"
    ):
      self.action_ctx.get_action_values()

  def test_get_action_values_setpoint_name_not_found_raises(self):
    self.action_ctx.setpoints[0].setpoint_name = "OOPS"
    with self.assertRaisesRegex(KeyError, "\\('air_handler_1', 'OOPS'\\)"):
      self.action_ctx.get_action_values()

  def test_get_hybrid_action(self):
    self.assertEqual(
        self.action_ctx.get_hybrid_action(),
        {
            "continuous_action": [-1.0, -1.0, -1.0],
            "discrete_action": [1.0, 1.0, 1.0],
        },
    )

  def test_setpoints_df(self):
    df = self.action_ctx.setpoints_df
    expected_df = pd.DataFrame([
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": (
                "air_handler_1_supply_air_heating_temperature_setpoint"
            ),
            "device_id": "air_handler_1",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "setpoint_value": 285.0,
            "setpoint_justification": "To cool the air.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": "air_handler_1_supervisor_run_command",
            "device_id": "air_handler_1",
            "setpoint_name": "supervisor_run_command",
            "setpoint_value": 1.0,
            "setpoint_justification": "To turn the device on.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": "boiler_1_supply_water_setpoint",
            "device_id": "boiler_1",
            "setpoint_name": "supply_water_setpoint",
            "setpoint_value": 310.0,
            "setpoint_justification": "To heat the water.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": "boiler_1_supervisor_run_command",
            "device_id": "boiler_1",
            "setpoint_name": "supervisor_run_command",
            "setpoint_value": 1.0,
            "setpoint_justification": "To turn the device on.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": (
                "air_handler_2_supply_air_heating_temperature_setpoint"
            ),
            "device_id": "air_handler_2",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "setpoint_value": 285.0,
            "setpoint_justification": "To cool the air.",
        },
        {
            "timestamp": "2025-01-01 12:00:00",
            "validity_interval": 60,
            "justification": "These are my overall goals.",
            "action_name": "air_handler_2_supervisor_run_command",
            "device_id": "air_handler_2",
            "setpoint_name": "supervisor_run_command",
            "setpoint_value": 1.0,
            "setpoint_justification": "To turn the device on.",
        },
    ])
    pd.testing.assert_frame_equal(df, expected_df)

  def test_flattened_setpoints_record(self):
    record = self.action_ctx.flattened_setpoints_record
    expected_record = {
        "timestamp": "2025-01-01 12:00:00",
        "validity_interval": 60,
        "justification": "These are my overall goals.",
        "air_handler_1_supervisor_run_command": 1.0,
        "air_handler_1_supervisor_run_command_justification": (
            "To turn the device on."
        ),
        "air_handler_2_supervisor_run_command": 1.0,
        "air_handler_2_supervisor_run_command_justification": (
            "To turn the device on."
        ),
        "boiler_1_supervisor_run_command": 1.0,
        "boiler_1_supervisor_run_command_justification": (
            "To turn the device on."
        ),
        "air_handler_1_supply_air_heating_temperature_setpoint": 285.0,
        "air_handler_1_supply_air_heating_temperature_setpoint_justification": (
            "To cool the air."
        ),
        "air_handler_2_supply_air_heating_temperature_setpoint": 285.0,
        "air_handler_2_supply_air_heating_temperature_setpoint_justification": (
            "To cool the air."
        ),
        "boiler_1_supply_water_setpoint": 310.0,
        "boiler_1_supply_water_setpoint_justification": "To heat the water.",
    }
    self.assertDictEqual(record, expected_record)


#
# CONSTRUCTOR TESTS
#


class ActionContextFromJsonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_environment(layout=env_conftest.DEMO_LAYOUT)
    self.schema = schema_conftest.create_action()
    self.json = self.schema.model_dump_json()
    self.action_ctx = action_context.ActionContext.from_json(
        txt=self.json, env=self.env, clip=True
    )

  def test_json(self):
    self.assertIsInstance(self.json, str)

  def test_initialization(self):
    self.assertIsInstance(self.action_ctx, action_context.ActionContext)

  def test_extra_attributes(self):
    self.assertIs(self.action_ctx.env, self.env)
    self.assertTrue(self.action_ctx.clip)
    self.assertEmpty(self.action_ctx.guardrails_exceeded)

  def test_schema_contents(self):
    with self.subTest("timestamp"):
      self.assertEqual(self.action_ctx.timestamp, self.schema.timestamp)

    with self.subTest("justification"):
      self.assertEqual(self.action_ctx.justification, self.schema.justification)

    with self.subTest("validity_interval"):
      self.assertEqual(
          self.action_ctx.validity_interval, self.schema.validity_interval
      )

    with self.subTest("setpoints"):
      self.assertEqual(self.action_ctx.setpoints, self.schema.setpoints)


class HybridActionContextFromJsonTest(ActionContextFromJsonTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.schema = schema_conftest.create_hybrid_action()
    self.json = self.schema.model_dump_json()
    self.action_ctx = action_context.HybridActionContext.from_json(
        txt=self.json, env=self.env, clip=True
    )

  def test_initialization(self):
    self.assertIsInstance(self.action_ctx, action_context.HybridActionContext)


#
# GUARDRAILS / VALIDATION TESTS
#


class ActionContextGuardrailsTest(absltest.TestCase):
  """Tests for guardrails behavior when clipping is disabled."""

  CLIPPING_ENABLED = False

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_environment(layout=env_conftest.DEMO_LAYOUT)
    self.schema = schema_conftest.create_action()
    self.clip = self.CLIPPING_ENABLED

  def test_device_id_not_found_raises(self):
    self.schema.setpoints[0].device_id = "OOPS"
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Setpoint for \\('OOPS', 'supply_air_heating_temperature_setpoint'\\)"
        " not found in the environment",
    ):
      action_context.ActionContext(
          env=self.env, clip=self.clip, **self.schema.model_dump()
      )

  def test_setpoint_name_not_found_raises(self):
    self.schema.setpoints[0].setpoint_name = "OOPS"
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Setpoint for \\('air_handler_1', 'OOPS'\\) not found in the"
        " environment",
    ):
      action_context.ActionContext(
          env=self.env, clip=self.clip, **self.schema.model_dump()
      )

  def test_normalizer_not_found_raises(self):
    with mock.patch.dict(self.env.action_normalizers, {}, clear=True):
      with self.assertRaisesRegex(
          pydantic.ValidationError,
          "Normalizer not found for setpoint:"
          " 'air_handler_1_supply_air_heating_temperature_setpoint'.",
      ):
        action_context.ActionContext(
            env=self.env, clip=self.clip, **self.schema.model_dump()
        )

  def test_missing_setpoint_raises(self):
    self.schema.setpoints.pop()
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "The following setpoints are expected by the environment but are"
        " missing from the schema:.*'boiler_1_supply_water_setpoint'",
    ):
      action_context.ActionContext(
          env=self.env, clip=self.clip, **self.schema.model_dump()
      )

  # TESTS WHERE CLIPPING OPTION IS RELEVANT

  def test_clipping_option(self):
    self.assertFalse(self.clip)

  def test_setpoint_value_above_range(self):
    self.schema.setpoints[0].setpoint_value = 300.0  # Above range
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        " Value 300.0 for setpoint \\('air_handler_1'.*"
        "'supply_air_heating_temperature_setpoint'\\) is outside expected"
        " range \\[285\\.0, 295\\.0\\]",
    ):
      action_context.ActionContext(
          env=self.env, clip=self.clip, **self.schema.model_dump()
      )

  def test_setpoint_value_below_range(self):
    self.schema.setpoints[0].setpoint_value = 200.0  # Below range
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        " Value 200.0 for setpoint \\('air_handler_1'.*"
        "'supply_air_heating_temperature_setpoint'\\) is outside expected"
        " range \\[285\\.0, 295\\.0\\]",
    ):
      action_context.ActionContext(
          env=self.env, clip=self.clip, **self.schema.model_dump()
      )


class ActionContextGuardrailsClippingTest(ActionContextGuardrailsTest):
  """Tests for guardrails behavior when clipping is enabled."""

  CLIPPING_ENABLED = True

  def test_clipping_option(self):
    self.assertTrue(self.clip)

  def test_setpoint_value_above_range(self):
    self.schema.setpoints[0].setpoint_value = 300.0  # Above range
    action_ctx = action_context.ActionContext(
        env=self.env, clip=self.clip, **self.schema.model_dump()
    )

    with self.subTest(name="clips_value_to_max"):
      self.assertEqual(action_ctx.setpoints[0].setpoint_value, 295.0)  # Max

    with self.subTest(name="logs_guardrails_error"):
      self.assertLen(action_ctx.guardrails_exceeded, 1)
      self.assertEqual(
          action_ctx.guardrails_exceeded[0],
          action_context.GuardrailsExceededRecord(
              device_id="air_handler_1",
              setpoint_name="supply_air_heating_temperature_setpoint",
              requested_value=300.0,
              setpoint_range=(285.0, 295.0),
              clipped_value=295.0,
          ),
      )

  def test_setpoint_value_below_range(self):
    self.schema.setpoints[0].setpoint_value = 200.0  # Below range
    action_ctx = action_context.ActionContext(
        env=self.env, clip=self.clip, **self.schema.model_dump()
    )

    with self.subTest(name="clips_value_to_min"):
      self.assertEqual(action_ctx.setpoints[0].setpoint_value, 285.0)  # Min

    with self.subTest(name="logs_guardrails_error"):
      self.assertLen(action_ctx.guardrails_exceeded, 1)
      self.assertEqual(
          action_ctx.guardrails_exceeded[0],
          action_context.GuardrailsExceededRecord(
              device_id="air_handler_1",
              setpoint_name="supply_air_heating_temperature_setpoint",
              requested_value=200.0,
              setpoint_range=(285.0, 295.0),
              clipped_value=285.0,
          ),
      )


#
# CUSTOM VALIDITY INTERVALS
#


class ActionContextWithCustomValidityIntervalsTest(absltest.TestCase):

  IS_HYBRID = False

  def setUp(self):
    super().setUp()
    self.custom_intervals = [15, 30, 45, 60]
    self.schema = action_context.create_action_context_model(
        custom_intervals=self.custom_intervals,
        hybrid=self.IS_HYBRID,
    )

  def test_initialization(self):
    self.assertTrue(issubclass(self.schema, action_context.ActionContext))
    self.assertFalse(
        issubclass(self.schema, action_context.HybridActionContext)
    )

  def test_validity_interval_options(self):
    self.assertCountEqual(
        get_args(self.schema.__annotations__["validity_interval"]),
        self.custom_intervals,
    )


class HybridActionContextWithCustomValidityIntervalsTest(
    ActionContextWithCustomValidityIntervalsTest
):

  IS_HYBRID = True

  def test_initialization(self):
    self.assertTrue(issubclass(self.schema, action_context.ActionContext))
    self.assertTrue(issubclass(self.schema, action_context.HybridActionContext))


#
# FACTORY FUNCTION TESTS
#


class ActionContextFactoryTest(absltest.TestCase):

  def test_defaults(self):
    action_ctx = schema_conftest.create_action_context()
    self.assertIsInstance(action_ctx, action_context.ActionContext)

  def test_overrides(self):
    env = env_conftest.create_environment(layout=env_conftest.DEMO_LAYOUT)
    action = schema_conftest.create_action()
    action.justification = "Custom justification."

    action_ctx = schema_conftest.create_action_context(env=env, action=action)
    self.assertIsInstance(action_ctx, action_context.ActionContext)
    self.assertEqual(action_ctx.justification, "Custom justification.")


class HybridActionContextFactoryTest(ActionContextFactoryTest):

  def test_defaults(self):
    action_ctx = schema_conftest.create_hybrid_action_context()
    self.assertIsInstance(action_ctx, action_context.HybridActionContext)

  def test_overrides(self):
    env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    action = schema_conftest.create_hybrid_action()
    action.justification = "Custom justification."

    action_ctx = schema_conftest.create_hybrid_action_context(
        env=env, action=action
    )
    self.assertIsInstance(action_ctx, action_context.HybridActionContext)
    self.assertEqual(action_ctx.justification, "Custom justification.")


if __name__ == "__main__":
  absltest.main()
