import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import default_agent
from smart_buildings.smart_control.llm.schema import action_context


class AgentEnvironmentValidationsTest(parameterized.TestCase):

  AGENT_CLASS = default_agent.DefaultPolicyAgent

  def setUp(self):
    super().setUp()
    self.env = mock.create_autospec(environment.Environment, instance=True)
    self.env.action_names = list(env_conftest.DEFAULT_ACTIONS.keys())
    self.env.default_policy_values = list(env_conftest.DEFAULT_ACTIONS.values())

  def test_valid_environment(self):
    agent = self.AGENT_CLASS(self.env)
    self.assertIsInstance(agent, self.AGENT_CLASS)

  def test_validate_action_names(self):
    self.env.action_names = None

    with self.assertRaisesRegex(
        ValueError, "Expecting environment to have action names."
    ):
      self.AGENT_CLASS(self.env)

  def test_validate_default_values(self):
    self.env.default_policy_values = None

    with self.assertRaisesRegex(
        ValueError, "Expecting environment to have default policy values."
    ):
      self.AGENT_CLASS(self.env)

  def test_validate_number_of_action_names_and_default_values(self):
    self.env.action_names = self.env.action_names[1:]

    with self.assertRaisesRegex(
        ValueError,
        "Expecting environment to have the same number of action names and"
        " default policy values.",
    ):
      self.AGENT_CLASS(self.env)


class DefaultAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.env = self._create_environment()
    self.agent = self._create_agent(self.env)

  def _create_environment(
      self, start_timestamp: pd.Timestamp | None = None
  ) -> environment.Environment:
    return env_conftest.create_environment(
        layout=env_conftest.DEMO_LAYOUT,
        default_actions=env_conftest.DEFAULT_ACTIONS,
        start_timestamp=start_timestamp,
    )

  def _create_agent(
      self, env: environment.Environment
  ) -> default_agent.DefaultPolicyAgent:
    return default_agent.DefaultPolicyAgent(env=env)

  def test_initialization(self):
    self.assertIsInstance(self.agent, default_agent.DefaultPolicyAgent)

  def test_environment(self):
    self.assertIsInstance(self.agent.env, environment.Environment)

  def test_json_metadata(self):
    self.assertEqual(
        self.agent.json_metadata,
        {
            "type": "DefaultPolicyAgent",
            "default_policy": {
                "action_names": self.env.action_names,
                "default_values": self.env.default_action_values,
            },
            "clip": True,
        },
    )

  def test_json_metadata_is_serializable(self):
    self.assertEqual(
        self.agent.json_metadata,
        json.loads(json.dumps(self.agent.json_metadata, indent=2)),
    )

  def test_default_action_context(self):
    ctx = self.agent.get_default_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)

    with self.subTest(name="timestamp"):
      self.assertEqual(ctx.timestamp, str(self.env.current_local_timestamp))

    with self.subTest(name="justification"):
      self.assertEqual(ctx.justification, default_agent.DEFAULT_JUSTIFICATION)

    with self.subTest(name="validity_interval"):
      self.assertEqual(ctx.validity_interval, self.env.time_step_mins)

    with self.subTest(name="setpoints"):
      self.assertLen(ctx.setpoints, len(self.env.action_names))

      # Setpoint and device names should match the env's action names:
      names = [(sp.device_id, sp.setpoint_name) for sp in ctx.sorted_setpoints]
      self.assertEqual(
          names,
          [
              ("air_handler_1", "supply_air_heating_temperature_setpoint"),
              ("boiler_1", "supply_water_setpoint"),
              ("air_handler_2", "supply_air_heating_temperature_setpoint"),
          ],
      )

      # Setpoint values should be native versions of the env's default values:
      setpoint_values = [sp.setpoint_value for sp in ctx.setpoints]
      self.assertSequenceAlmostEqual(setpoint_values, [290.0, 310.0, 290.0])

  def test_get_action_context(self):
    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)
    self.assertEqual(ctx, self.agent.get_default_action_context())


class DefaultHybridActionAgentTest(DefaultAgentTest):

  def _create_environment(
      self, start_timestamp: pd.Timestamp | None = None
  ) -> hybrid_action_environment.HybridActionEnvironment:
    return env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
        start_timestamp=start_timestamp,
    )

  def test_environment(self):
    self.assertIsInstance(
        self.agent.env, hybrid_action_environment.HybridActionEnvironment
    )

  def test_default_action_context(self):
    ctx = self.agent.get_default_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)

    with self.subTest(name="timestamp"):
      self.assertEqual(ctx.timestamp, str(self.env.current_local_timestamp))

    with self.subTest(name="justification"):
      self.assertEqual(ctx.justification, default_agent.DEFAULT_JUSTIFICATION)

    with self.subTest(name="validity_interval"):
      self.assertEqual(ctx.validity_interval, self.env.time_step_mins)

    with self.subTest(name="setpoints"):
      self.assertLen(ctx.setpoints, len(self.env.action_names))
      self.assertSequenceAlmostEqual(
          ctx.get_action_values(), self.env.default_action_values
      )

    # Setpoint and device names should match the env's action names:
    names = [(sp.device_id, sp.setpoint_name) for sp in ctx.sorted_setpoints]
    with self.subTest(name="setpoint_names"):
      self.assertEqual(
          names,
          [
              ("air_handler_1", "supply_air_heating_temperature_setpoint"),
              ("air_handler_1", "supervisor_run_command"),
              ("boiler_1", "supply_water_setpoint"),
              ("boiler_1", "supervisor_run_command"),
              ("air_handler_2", "supply_air_heating_temperature_setpoint"),
              ("air_handler_2", "supervisor_run_command"),
          ],
      )

    # Setpoint values should be native versions of the env's default values:
    setpoint_values = [sp.setpoint_value for sp in ctx.setpoints]
    with self.subTest(name="setpoint_values"):
      self.assertSequenceAlmostEqual(
          setpoint_values, [290.0, 0, 310.0, 0, 290.0, 0]
      )

  def test_get_action_context(self):
    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)
    self.assertEqual(ctx, self.agent.get_default_action_context())


if __name__ == "__main__":
  absltest.main()
