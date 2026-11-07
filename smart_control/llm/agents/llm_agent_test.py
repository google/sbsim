import json
import time
from typing import get_args
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pydantic
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import llm_agent
from smart_buildings.smart_control.llm.agents import schedule_agent_test
from smart_buildings.smart_control.llm.prompts import promptmaker as pm
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.schema import conftest as schema_conftest
from smart_buildings.smart_control.llm.schema import output_schema
from smart_buildings.smart_control.llm.services import llm_service
from smart_buildings.smart_control.llm.utils import schedule_tool as schedule_lib
from smart_buildings.smart_control.llm.utils import schedule_tool_test


class TextParsingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='json_fence', input_text='```json\n{"a": 1}\n```'),
      dict(testcase_name='plain_fence', input_text='```\n{"a": 1}\n```'),
      dict(testcase_name='no_fences', input_text='{"a": 1}'),
      dict(
          testcase_name='text_around_fences',
          input_text='Text\n```json\n{"a": 1}\n```\nMore text',
      ),
  )
  def test_parse_response_text_variants(self, input_text):
    expected_text = json.dumps({'a': 1})
    self.assertEqual(llm_agent.parse_response_text(input_text), expected_text)

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_json_valid_schema',
          input_text=schema_conftest.create_hybrid_action_response(),
      ),
      dict(
          testcase_name='valid_json_empty_setpoints',
          input_text=schema_conftest.create_hybrid_action_response(
              empty_setpoints=True
          ),
      ),
      dict(
          testcase_name='valid_json_missing_setpoint',
          input_text=schema_conftest.create_hybrid_action_response(
              missing_setpoint=True
          ),
      ),
      dict(
          testcase_name='valid_json_missing_field',
          input_text=schema_conftest.create_hybrid_action_response(
              missing_field=True
          ),
      ),
  )
  def test_parse_response_text_valid_json_invalid_schema(self, input_text):
    self.assertEqual(llm_agent.parse_response_text(input_text), input_text)

  def test_parse_response_text_non_string_input(self):
    with self.assertRaisesRegex(ValueError, 'Expecting a string response'):
      llm_agent.parse_response_text(None)


class LLMAgentTest(
    schedule_agent_test.ScheduleHybridActionAgentTest, parameterized.TestCase
):

  env: hybrid_action_environment.HybridActionEnvironment
  agent: llm_agent.LLMAgent
  mock_llm_service: mock.MagicMock
  schedule_tool: schedule_lib.BuildingScheduleTool

  def _create_agent(
      self, env: hybrid_action_environment.HybridActionEnvironment
  ) -> llm_agent.LLMAgent:
    self.mock_llm_service = mock.create_autospec(
        llm_service.BaseLLMService, instance=True, spec_set=True
    )
    self.mock_llm_service.json_metadata = {'type': 'MockLLMService'}
    self.schedule_tool = schedule_lib.BuildingScheduleTool(env=env)
    return llm_agent.LLMAgent(
        env=env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        schedule_tool=self.schedule_tool,
    )

  def test_json_metadata(self):
    self.assertEqual(
        self.agent.json_metadata,
        {
            'type': 'LLMAgent',
            'default_policy': {
                'action_names': [
                    'air_handler_1_supply_air_heating_temperature_setpoint',
                    'air_handler_1_supervisor_run_command',
                    'boiler_1_supply_water_setpoint',
                    'boiler_1_supervisor_run_command',
                    'air_handler_2_supply_air_heating_temperature_setpoint',
                    'air_handler_2_supervisor_run_command',
                ],
                'default_values': [0.0, -1.0, -1.0, -1.0, 0.0, -1.0],
            },
            'override_discrete_defaults': True,
            'schedule_policy': schedule_tool_test.SCHEDULE_METADATA,
            'llm_service': {'type': 'MockLLMService'},
            'promptmaker': {
                'type': 'Promptmaker',
                'output_schema_class': 'HybridActionContext',
                'include_weights': True,
                'occupancy_mode_min': 10,
                'temp_display_unit': 'Fahrenheit',
                'building_info': {
                    'stories': 'two',
                    'sqft': 96_000,
                    'location': 'Mountain View, California',
                    'name': 'SB-1',
                },
            },
            'output_schema': {'type': 'HybridActionContext'},
            'temp_display_unit': 'Fahrenheit',
            'max_tries': 5,
            'clip': True,
        },
    )

  # PROMPT

  def test_make_prompt(self):
    observation_response = self.env.get_observation_response()
    reward_info = self.env.get_reward_info()
    prompt = self.agent.make_prompt(observation_response, reward_info)
    self.assertIsInstance(prompt, str)
    self.assertNotEmpty(prompt)

  # RESPONSE VALIDATION

  def test_validate_action_context(self):
    valid_response_json = schema_conftest.create_hybrid_action_response()
    ctx = self.agent.validate_action_context(valid_response_json)
    self.assertIsInstance(ctx, action_context.ActionContext)

  def test_validate_action_context_invalid_json(self):
    with self.assertRaisesRegex(json.JSONDecodeError, 'Expecting value'):
      self.agent.validate_action_context('oops, invalid json')

  def test_validate_action_context_invalid_schema(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError, r'validity_interval\n\s+Field required'
    ):
      self.agent.validate_action_context('{"valid": "json"}')

  def test_validate_action_context_missing_setpoint(self):
    invalid_schema_json = schema_conftest.create_hybrid_action_response(
        missing_setpoint=True
    )
    with self.assertRaisesRegex(
        pydantic.ValidationError, r'missing from the schema'
    ):
      self.agent.validate_action_context(invalid_schema_json)

  def test_validate_action_context_guardrails_exceeded(self):
    invalid_schema_json = schema_conftest.create_hybrid_action_response(
        ahu_1_run_command=1,
        ahu_1_supply_air_temp=99999.0,
        hws_run_command=1,
        hws_supply_water_temp=99999.0,
    )
    ctx = self.agent.validate_action_context(invalid_schema_json)
    self.assertEqual(
        ctx.guardrails_exceeded,
        [
            action_context.GuardrailsExceededRecord(
                device_id='air_handler_1',
                setpoint_name='supply_air_heating_temperature_setpoint',
                requested_value=99999.0,
                setpoint_range=(285.0, 295.0),
                clipped_value=295.0,
            ),
            action_context.GuardrailsExceededRecord(
                device_id='boiler_1',
                setpoint_name='supply_water_setpoint',
                requested_value=99999.0,
                setpoint_range=(310.0, 350.0),
                clipped_value=350.0,
            ),
        ],
    )

  # GET ACTION

  def test_get_action_context_success(self):
    valid_response_json = schema_conftest.create_hybrid_action_response()
    self.mock_llm_service.get_response.return_value = valid_response_json

    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)

    with self.subTest(name='calls_the_llm'):
      self.mock_llm_service.get_response.assert_called_once()

    with self.subTest(name='no_errors'):
      self.assertEmpty(self.agent.errors)

  def test_get_action_context_fenced_response(self):
    json_data = schema_conftest.create_hybrid_action_response()
    fenced_response = (
        f'Here is the JSON you requested:\n```json\n{json_data}\n```\nI hope'
        ' that helps!'
    )
    self.mock_llm_service.get_response.return_value = fenced_response

    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)

    with self.subTest(name='calls_the_llm'):
      self.mock_llm_service.get_response.assert_called_once()

    with self.subTest(name='no_errors'):
      self.assertEmpty(self.agent.errors)


class AlternativeSchemaTestBase(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
    )
    self.mock_llm_service = mock.create_autospec(
        llm_service.BaseLLMService, instance=True, spec_set=True
    )
    self.mock_llm_service.json_metadata = {'type': 'MockLLMService'}
    valid_json = schema_conftest.create_hybrid_action_response()
    self.mock_llm_service.get_response.return_value = valid_json


class LLMAgentSetpointsActionSchemaTest(AlternativeSchemaTestBase):

  def setUp(self):
    super().setUp()
    self.agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
    )

  def test_get_action_context(self):
    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)


class LLMAgentCustomValidityIntervalTest(AlternativeSchemaTestBase):

  def setUp(self):
    self.custom_intervals = [15, 30, 60, 90]
    super().setUp()
    self.output_schema_class = action_context.create_action_context_model(
        custom_intervals=self.custom_intervals
    )
    self.promptmaker = pm.Promptmaker(
        env=self.env,
        output_schema_class=self.output_schema_class,
        lazy_init_protos=True,
    )
    self.agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker=self.promptmaker,
    )

  def test_custom_validity_interval(self):
    self.assertEqual(
        get_args(
            self.agent.output_schema_class.__annotations__['validity_interval']
        ),
        tuple(self.custom_intervals),
    )

    with self.subTest(name='in_prompt'):
      prompt = self.agent.make_prompt(
          self.env.get_observation_response(), self.env.get_reward_info()
      )
      self.assertIn(str(self.custom_intervals), prompt)

  def test_get_action_context(self):
    valid_json = schema_conftest.create_hybrid_action_response()
    self.mock_llm_service.get_response.return_value = valid_json

    ctx = self.agent.get_action_context()
    self.assertIsInstance(ctx, action_context.ActionContext)


class LLMAgentPromptmakerValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_llm_service = mock.create_autospec(
        llm_service.BaseLLMService, instance=True, spec_set=True
    )
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
    )
    self.promptmaker_instance = pm.Promptmaker(
        env=self.env, lazy_init_protos=True
    )

  def test_init_with_promptmaker_instance_succeeds(self):
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker=self.promptmaker_instance,
    )
    self.assertIsInstance(agent.promptmaker, pm.Promptmaker)
    self.assertEqual(agent.promptmaker, self.promptmaker_instance)

  def test_init_with_promptmaker_class_succeeds(self):
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
    )
    self.assertIsInstance(agent.promptmaker, pm.Promptmaker)

    with self.subTest(name='uses_default_arguments'):
      self.assertEqual(
          agent.promptmaker.output_schema_class,
          action_context.HybridActionContext,
      )
      self.assertTrue(agent.promptmaker.lazy_init_protos)
      self.assertTrue(agent.promptmaker.include_weights)

  def test_init_raises_error_with_neither_promptmaker_nor_class(self):
    with self.assertRaisesRegex(
        ValueError,
        'Either a promptmaker instance or class must be provided, not both.',
    ):
      llm_agent.LLMAgent(
          env=self.env,
          llm_service=self.mock_llm_service,
          promptmaker=None,
          promptmaker_class=None,
      )

  def test_init_raises_error_with_both_promptmaker_and_class(self):
    with self.assertRaisesRegex(
        ValueError,
        'Either a promptmaker instance or class must be provided, not both.',
    ):
      llm_agent.LLMAgent(
          env=self.env,
          llm_service=self.mock_llm_service,
          promptmaker=self.promptmaker_instance,
          promptmaker_class=pm.Promptmaker,
      )


class LLMAgentRetryTest(LLMAgentTest):
  """Tests for retry and backoff behavior.

  NOTE: We need to mock time.sleep because of the retry logic used in LLMAgent,
  which uses the backoff library to handle retries when the LLM fails to return
  a valid response. By default, backoff attempts to wait between retries by
  calling time.sleep. If we do not mock time.sleep, the unit tests will actually
  pause and wait, but instead we are mocking it to return immediately.
  """

  @mock.patch.object(time, 'sleep', return_value=None)
  def test_retry_succeeds_after_failures(self, mock_sleep):
    valid_response_json = schema_conftest.create_hybrid_action_response()
    self.mock_llm_service.get_response.side_effect = [
        RuntimeError('Service Fail 1'),  # on_backoff records this
        'oops invalid json',  # on_backoff records this
        valid_response_json,
    ]
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        max_tries=3,
    )

    ctx = agent.get_action_context()

    # The agent should succeed on the 3rd attempt.
    self.assertIsInstance(ctx, action_context.ActionContext)
    self.assertEqual(self.mock_llm_service.get_response.call_count, 3)

    # 2 errors should be recorded by on_backoff.
    self.assertLen(agent.errors, 2)
    self.assertEqual(agent.errors[0].error_type, 'RuntimeError')
    self.assertEqual(agent.errors[0].metadata['tries'], 1)
    self.assertEqual(agent.errors[1].error_type, 'JSONDecodeError')
    self.assertEqual(agent.errors[1].metadata['tries'], 2)

  @mock.patch.object(time, 'sleep', return_value=None)
  def test_exceeds_max_retries_and_falls_back(self, mock_sleep):
    self.mock_llm_service.get_response.side_effect = RuntimeError(
        'Always failing'
    )
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        max_tries=2,
    )
    # Mock scheduled action to confirm fallback.
    scheduled_ctx = mock.MagicMock()
    with mock.patch.object(
        agent,
        'get_scheduled_action_context',
        return_value=scheduled_ctx,
        autospec=True,
    ) as mock_get_scheduled:
      ctx = agent.get_action_context()

      # Check that fallback occurred.
      self.assertEqual(ctx, scheduled_ctx)
      mock_get_scheduled.assert_called_once()

    # on_backoff is called for try 1, on_giveup for try 2.
    # Total calls: 2.
    # Total errors:
    # 1st recorded by _on_backoff
    # 2nd recorded by _on_giveup
    # MaxRetriesExceededError recorded by _on_giveup
    self.assertEqual(self.mock_llm_service.get_response.call_count, 2)
    self.assertLen(agent.errors, 3)

    # Error from on_backoff
    self.assertEqual(agent.errors[0].error_type, 'RuntimeError')
    self.assertEqual(agent.errors[0].metadata['tries'], 1)
    self.assertIsNotNone(agent.errors[0].metadata['wait'])

    # Error from on_giveup
    self.assertEqual(agent.errors[1].error_type, 'RuntimeError')
    self.assertEqual(agent.errors[1].metadata['tries'], 2)
    self.assertIsNone(agent.errors[1].metadata['wait'])

    # Exhaustion error from on_giveup
    self.assertEqual(agent.errors[2].error_type, 'MaxRetriesExceededError')

  @mock.patch.object(time, 'sleep', return_value=None)
  def test_pydantic_error_details_on_giveup(self, mock_sleep):
    invalid_schema_json = json.dumps({'validity_interval': 15, 'setpoints': []})
    self.mock_llm_service.get_response.return_value = invalid_schema_json
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        max_tries=1,
    )

    agent.get_action_context()

    # on_giveup is called for try 1 because max_tries=1.
    self.assertLen(agent.errors, 2)
    err = agent.errors[0]
    self.assertEqual(err.error_type, 'ValidationError')
    self.assertEqual(err.metadata['tries'], 1)

    # Check that pydantic error details were recorded.
    self.assertIsInstance(err.details, list)
    self.assertNotEmpty(err.details)
    self.assertEqual(err.details[0]['type'], 'missing')
    self.assertEqual(err.details[0]['loc'], ('timestamp',))

  @mock.patch.object(llm_agent.logging, 'exception')
  @mock.patch.object(time, 'sleep', return_value=None)
  def test_exceeds_max_retries_falls_back_to_previous_action(
      self, mock_sleep, mock_exception
  ):
    valid_response_json = schema_conftest.create_hybrid_action_response(
        validity_interval=15
    )
    self.mock_llm_service.get_response.side_effect = [
        valid_response_json,  # First call succeeds
        RuntimeError('Always failing'),  # Second call fails
        RuntimeError('Always failing'),  # Third call fails (exceeds max_tries)
    ]
    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        max_tries=2,
    )

    # 1. First call to get_action_context succeeds.
    ctx1 = agent.get_action_context()
    self.assertEqual(ctx1.validity_interval, 15)

    # 2. Second call to get_action_context fails all retries, should fallback to
    # the previous successful action with the environment's time step interval.
    ctx2 = agent.get_action_context()
    mock_exception.assert_called_once_with(
        'LLM MAX TRIES EXCEEDED. FALLING BACK TO PREVIOUS LLM ACTION...'
    )

    self.assertEqual(ctx2.validity_interval, agent.env.time_step_mins)
    self.assertEqual(ctx2.setpoints, ctx1.setpoints)
    self.assertEqual(
        ctx2.justification, 'Previous LLM action (max retries exceeded)'
    )
    self.assertEqual(ctx2.timestamp, ctx1.timestamp)

  @mock.patch.object(llm_agent.logging, 'exception')
  @mock.patch.object(
      llm_agent.LLMAgent, 'get_scheduled_action_context', autospec=True
  )
  @mock.patch.object(time, 'sleep', return_value=None)
  def test_exceeds_max_retries_no_previous_action_falls_back_to_schedule(
      self, mock_sleep, mock_get_scheduled, mock_exception
  ):
    self.mock_llm_service.get_response.side_effect = RuntimeError('OOPS')

    agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
        max_tries=2,
    )

    scheduled_ctx = mock.MagicMock()
    mock_get_scheduled.return_value = scheduled_ctx

    ctx = agent.get_action_context()
    self.assertEqual(ctx, scheduled_ctx)
    mock_get_scheduled.assert_called_once()
    mock_exception.assert_called_once_with(
        'LLM MAX TRIES EXCEEDED. NO PREVIOUS LLM ACTION AVAILABLE. FALLING'
        ' BACK TO SCHEDULED ACTION...'
    )


class LLMAgentNonActionContextSchemaTest(AlternativeSchemaTestBase):

  def setUp(self):
    super().setUp()
    self.agent = llm_agent.LLMAgent(
        env=self.env,
        llm_service=self.mock_llm_service,
        promptmaker_class=pm.Promptmaker,
    )

  def test_validate_action_context_non_subclass(self):
    self.agent.output_schema_class = output_schema.SetpointsAction
    valid_response_json = schema_conftest.create_hybrid_action_response()

    ctx = self.agent.validate_action_context(valid_response_json)

    self.assertIsInstance(ctx, action_context.ActionContext)


if __name__ == '__main__':
  absltest.main()
