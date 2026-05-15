from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.agents import default_agent_test
from smart_buildings.smart_control.llm.agents import schedule_agent
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.utils import schedule_tool
from smart_buildings.smart_control.llm.utils import schedule_tool_test

TIME_ZONE = "US/Pacific"


class ScheduleHybridActionAgentTest(
    default_agent_test.DefaultHybridActionAgentTest
):

  agent: schedule_agent.SchedulePolicyAgent

  def _create_environment(
      self, start_timestamp=schedule_tool_test.CURRENT_LOCAL_TIMESTAMP
  ):
    return super()._create_environment(start_timestamp=start_timestamp)

  def _create_agent(self, env):
    return schedule_agent.SchedulePolicyAgent(env=env)

  def test_json_metadata(self):
    self.assertEqual(
        self.agent.json_metadata,
        {
            "type": "SchedulePolicyAgent",
            "default_policy": {
                "action_names": [
                    "air_handler_1_supply_air_heating_temperature_setpoint",
                    "air_handler_1_supervisor_run_command",
                    "boiler_1_supply_water_setpoint",
                    "boiler_1_supervisor_run_command",
                    "air_handler_2_supply_air_heating_temperature_setpoint",
                    "air_handler_2_supervisor_run_command",
                ],
                "default_values": [0.0, -1.0, -1.0, -1.0, 0.0, -1.0],
            },
            "clip": True,
            "override_discrete_defaults": True,
            "schedule_policy": schedule_tool_test.SCHEDULE_METADATA,
        },
    )

  def test_building_is_operational(self):
    self.assertTrue(self.agent.building_is_operational)

  def test_building_operational_mode(self):
    self.assertEqual(
        self.agent.building_operational_mode,
        schedule_tool.BuildingOperationalMode.ON,
    )

  def test_justifications(self):
    self.assertEqual(
        self.agent.scheduled_justification,
        "Scheduled action (ON)",
    )
    self.assertEqual(
        self.agent.scheduled_setpoint_justification,
        "Scheduled value (ON)",
    )

  def test_scheduled_action_context(self):
    ctx = self.agent.get_scheduled_action_context()
    self.assertIsInstance(ctx, action_context.HybridActionContext)

    with self.subTest(name="timestamp"):
      self.assertEqual(ctx.timestamp, str(self.env.current_local_timestamp))

    with self.subTest(name="justification"):
      self.assertEqual(ctx.justification, self.agent.scheduled_justification)

    with self.subTest(name="validity_interval"):
      self.assertEqual(ctx.validity_interval, self.env.time_step_mins)

    with self.subTest(name="setpoints"):
      self.assertLen(ctx.setpoints, len(self.env.action_names))

      # Setpoint and device names should match the env's action names:
      names = [(sp.device_id, sp.setpoint_name) for sp in ctx.setpoints]
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

      # Setpoint values should be native versions of the env's default values
      # overridden by schedule policy.
      setpoint_values = [sp.setpoint_value for sp in ctx.setpoints]
      self.assertSequenceAlmostEqual(
          setpoint_values, [290.0, 1.0, 310.0, 1.0, 290.0, 1.0]
      )

  def test_get_action_context(self):
    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.HybridActionContext)
    self.assertEqual(ctx, self.agent.get_scheduled_action_context())


class ScheduleScenariosTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
        start_timestamp=schedule_tool_test.CURRENT_LOCAL_TIMESTAMP,
    )

  @parameterized.parameters(
      # Parameters:
      # setpoint_name, native_val, is_operational, will_override, expected_val
      # ...
      # Continuous setpoint, building is operational:
      ("supply_air_heating_temperature_setpoint", 290.0, True, True, 290.0),
      ("supply_air_heating_temperature_setpoint", 290.0, True, False, 290.0),
      # Continuous setpoint, building is non-operational:
      ("supply_air_heating_temperature_setpoint", 290.0, False, True, 290.0),
      ("supply_air_heating_temperature_setpoint", 290.0, False, False, 290.0),
      # Discrete action, building is non-operational
      ("supervisor_run_command", 1.0, False, True, 0.0),
      ("supervisor_run_command", 1.0, False, False, 0.0),
      # Discrete action, building is operational, override defaults:
      ("supervisor_run_command", 1.0, True, True, 1.0),  # FLIPPED ON
      ("supervisor_run_command", 0.0, True, True, 1.0),  # FLIPPED ON
      ("supervisor_run_command", -1.0, True, True, 1.0),  # FLIPPED ON
      # Discrete action, building is operational, do not override defaults:
      ("supervisor_run_command", 1.0, True, False, 1.0),
      ("supervisor_run_command", 0.0, True, False, 0.0),
      ("supervisor_run_command", -1.0, True, False, -1.0),
  )
  def test_get_scheduled_native_value(
      self,
      setpoint_name,
      native_value,
      building_is_operational,
      override_discrete_defaults,
      expected_value,
  ):
    mock_schedule_tool = mock.Mock()
    type(mock_schedule_tool).building_is_operational = mock.PropertyMock(
        return_value=building_is_operational
    )
    agent = schedule_agent.SchedulePolicyAgent(
        env=self.env,
        schedule_tool=mock_schedule_tool,
        override_discrete_defaults=override_discrete_defaults,
    )
    self.assertEqual(
        agent.get_scheduled_native_value(setpoint_name, native_value),
        expected_value,
    )


if __name__ == "__main__":
  absltest.main()
