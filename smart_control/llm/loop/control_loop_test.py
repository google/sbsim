from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd
# pylint: disable=g-bad-import-order local package imports in their own section below third party packages
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import default_agent
from smart_buildings.smart_control.llm.loop import conftest
from smart_buildings.smart_control.llm.loop import control_loop
from smart_buildings.smart_control.utils import writer_lib
from tf_agents.trajectories import time_step as ts

CLOCK_TIMESTAMP = pd.Timestamp('2026-03-26 12:00:00')
EXAMPLE_TIME_STEP = ts.TimeStep(
    step_type=ts.StepType.MID,
    reward=np.array([10.0]),
    discount=np.array(1.0),
    observation=(),
)


class ClockTimestampTest(absltest.TestCase):

  def test_get_clock_timestamp(self):
    with mock.patch.object(
        pd.Timestamp, 'now', return_value=CLOCK_TIMESTAMP, autospec=True
    ):
      self.assertEqual(
          control_loop.get_clock_timestamp(),
          CLOCK_TIMESTAMP,
      )


class TimestampParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.time_zone = 'US/Pacific'

  def test_parse_timestamp_without_time_zone_localizes(self):
    timestamp = pd.Timestamp('2025-12-12 00:00:00')
    self.assertIsNone(timestamp.tzinfo)
    self.assertEqual(
        control_loop.parse_timestamp(timestamp, self.time_zone),
        pd.Timestamp('2025-12-12 00:00:00', tz=self.time_zone),
    )

  def test_parse_timestamp_with_different_time_zone_converts(self):
    timestamp = pd.Timestamp('2025-12-12 00:00:00', tz='UTC')
    self.assertIsNotNone(timestamp.tzinfo)
    self.assertEqual(
        control_loop.parse_timestamp(timestamp, self.time_zone),
        pd.Timestamp('2025-12-11 16:00:00', tz=self.time_zone),
    )

  def test_parse_timestamp_with_same_time_zone_remains_the_same(self):
    timestamp = pd.Timestamp('2025-12-12 00:00:00', tz=self.time_zone)
    self.assertIsNotNone(timestamp.tzinfo)
    self.assertEqual(
        control_loop.parse_timestamp(timestamp, self.time_zone),
        pd.Timestamp('2025-12-12 00:00:00', tz=self.time_zone),
    )


class MetricsWriterValidationTest(absltest.TestCase):

  def _create_loop(
      self, writer: writer_lib.BaseWriter
  ) -> control_loop.ControlLoop:
    env = env_conftest.create_hybrid_action_environment(
        # writer_factory=lambda metrics_path: writer,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
    )
    agent = default_agent.DefaultPolicyAgent(env=env)
    env._metrics_writer = writer
    return control_loop.ControlLoop(agent=agent)

  def test_metrics_writer_with_valid_interface(self):
    writer = mock.create_autospec(writer_lib.BaseWriter, instance=True)
    self.assertTrue(hasattr(writer, 'output_dir'))
    self.assertTrue(hasattr(writer, 'write_json'))

    loop = self._create_loop(writer=writer)
    self.assertEqual(loop.metrics_writer, writer)

  def test_writer_without_output_dir_raises_error(self):
    writer = mock.create_autospec(writer_lib.BaseWriter, instance=True)
    del writer.output_dir

    with self.assertRaisesRegex(
        ValueError, 'Metrics writer does not have output_dir attribute.'
    ):
      self._create_loop(writer=writer)

  def test_writer_without_write_json_method_raises_error(self):
    writer = mock.create_autospec(writer_lib.BaseWriter, instance=True)
    del writer.write_json

    with self.assertRaisesRegex(
        ValueError, 'Metrics writer does not have write_json method.'
    ):
      self._create_loop(writer=writer)


class LoopTest(absltest.TestCase):
  """Tests for the setup of the control loop, before it has run."""

  def setUp(self):
    super().setUp()
    self.loop = conftest.create_loop(max_steps=5)

  def test_initialization(self):
    self.assertIsInstance(self.loop, control_loop.ControlLoop)

  def test_agent(self):
    self.assertIsInstance(self.loop.agent, default_agent.DefaultPolicyAgent)

  def test_env(self):
    self.assertIsInstance(
        self.loop.env, hybrid_action_environment.HybridActionEnvironment
    )

  def test_attributes(self):
    with self.subTest(name='max_steps'):
      self.assertEqual(self.loop.max_steps, 5)

    with self.subTest(name='cum_reward'):
      self.assertEqual(self.loop.cum_reward, 0.0)

  # ENVIRONMENT ATTRIBUTES

  def test_timestamps(self):
    with self.subTest(name='start_timestamp'):
      self.assertEqual(
          self.loop.start_timestamp,
          pd.Timestamp('2025-12-12 00:00:00', tz='US/Pacific'),
      )

    with self.subTest(name='end_timestamp'):
      self.assertEqual(
          self.loop.end_timestamp,
          pd.Timestamp('2025-12-15 00:00:00', tz='US/Pacific'),
      )

    with self.subTest(name='current_local_timestamp'):
      self.assertEqual(
          self.loop.current_local_timestamp,
          self.loop.env.current_local_timestamp,
      )

  def test_step_attributes(self):
    with self.subTest(name='days_per_episode'):
      self.assertEqual(self.loop.days_per_episode, 3)

    with self.subTest(name='time_step_interval'):
      self.assertEqual(self.loop.time_step_interval, pd.Timedelta(minutes=5))

    with self.subTest(name='steps_per_day'):
      self.assertEqual(self.loop.steps_per_day, 288)

    with self.subTest(name='steps_per_episode'):
      self.assertEqual(self.loop.steps_per_episode, 864)

    with self.subTest(name='episode_has_ended'):
      self.assertFalse(self.loop.episode_has_ended)

    with self.subTest(name='current_step'):
      self.assertEqual(self.loop.current_step, 0)

  # METRICS

  def test_metrics_output_dir(self):
    self.assertEqual(
        self.loop.metrics_output_dir, self.loop.metrics_writer.output_dir
    )

  def test_write_metadata(self):
    self.loop.env.metrics_writer.reset_mock()
    self.loop.write_metadata()
    self.loop.env.metrics_writer.write_json.assert_called_once_with(
        self.loop.json_metadata, 'metadata.json'
    )

  def test_write_results(self):
    self.loop.env.metrics_writer.reset_mock()
    with mock.patch.object(
        control_loop,
        'get_clock_timestamp',
        return_value=CLOCK_TIMESTAMP,
        autospec=True,
    ):
      self.loop.write_results()
      self.loop.env.metrics_writer.write_json.assert_called_once_with(
          self.loop.json_results, 'results.json'
      )

  def test_json_metadata(self):
    self.assertEqual(
        self.loop.json_metadata,
        {
            'start_timestamp': '2025-12-12 00:00:00-08:00',
            'end_timestamp': '2025-12-15 00:00:00-08:00',
            'days_per_episode': 3,
            'time_step_mins': 5,
            'steps_per_episode': 864,
            'env': self.loop.env.json_metadata,
            'agent': self.loop.agent.json_metadata,
        },
    )


class LoopResultsTest(absltest.TestCase):
  """Tests for the results of the control loop, after it has run."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.loop = conftest.create_loop(max_steps=5)

    # RUN THE LOOP (SO WE CAN TEST THE RESULTS AFTERWARDS)

    original_step_function = cls.loop.env.step

    def step_side_effect(*args, **kwargs):
      time_step = original_step_function(*args, **kwargs)
      return time_step._replace(reward=np.array([10.0]))

    with mock.patch.object(
        cls.loop.env, 'step', side_effect=step_side_effect, autospec=True
    ), mock.patch.object(
        control_loop,
        'get_clock_timestamp',
        autospec=True,
    ) as mock_clock_timestamp:
      mock_clock_timestamp.return_value = CLOCK_TIMESTAMP
      cls.loop.run()

  def test_json_results(self):
    with mock.patch.object(
        control_loop,
        'get_clock_timestamp',
        return_value=CLOCK_TIMESTAMP,
        autospec=True,
    ):
      self.assertEqual(
          self.loop.json_results,
          {
              'clock_timestamp': '2026-03-26 12:00:00',
              'current_timestamp': '2025-12-12 00:25:00-08:00',
              'current_step': 5,
              'cum_reward': 50.0,
              'results': [],
          },
      )


class LoopEndsWhenEpisodeEndsTest(absltest.TestCase):
  """Tests that the loop stops when episode has ended."""

  def test_stops_when_episode_has_ended(self):
    loop = conftest.create_loop(max_steps=None)
    with mock.patch.object(
        control_loop.ControlLoop,
        'episode_has_ended',
        new_callable=mock.PropertyMock,
        side_effect=[False, False, True],
    ) as mock_ended:
      loop.run()

    self.assertEqual(mock_ended.call_count, 3)
    self.assertEqual(loop.current_step, 2)


class ActionRejectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.loop = conftest.create_loop(max_steps=1)

  def test_action_rejection_reward(self):
    self.assertEqual(control_loop.ACTION_REJECTION_REWARD, -np.inf)

  def test_action_rejected_returns_true_when_reward_is_neg_inf(self):
    time_step = ts.TimeStep(
        step_type=ts.StepType.MID,
        reward=np.array([control_loop.ACTION_REJECTION_REWARD]),
        discount=np.array(1.0),
        observation=(),
    )
    self.assertTrue(self.loop._action_rejected(time_step))

  def test_action_rejected_returns_false_when_reward_is_not_neg_inf(self):
    self.assertFalse(self.loop._action_rejected(EXAMPLE_TIME_STEP))


class IntervalTest(absltest.TestCase):

  def test_validity_interval(self):
    loop = conftest.create_loop(max_steps=5)
    action_ctx = mock.Mock()
    action_ctx.validity_interval = 10  # minutes
    action_ctx.get_action.return_value = env_conftest.DEFAULT_HYBRID_ACTIONS

    # All this mocking and patching helps the environment step very fast, to
    # drastically reduce the time it takes to run this test.
    def step_side_effect(*args, **kwargs):
      del args, kwargs  # Unused.
      loop.env._step_count += 1
      return EXAMPLE_TIME_STEP

    with mock.patch.object(
        loop.agent,
        'get_action_context',
        return_value=action_ctx,
        autospec=True,
    ) as mock_get_action_context:
      with mock.patch.object(
          loop.env,
          'step',
          side_effect=step_side_effect,
          autospec=True,
      ) as mock_step:
        with mock.patch.object(
            loop.env,
            'get_observation_response',
            return_value=mock.Mock(),
            autospec=True,
        ):
          with mock.patch.object(
              loop.env,
              'get_reward_info_and_response',
              return_value=(mock.Mock(), mock.Mock()),
              autospec=True,
          ):
            loop.run()

    # The agent provides an initial action before the first step.
    # The environment is stepped five times, once every five minutes, for a
    # total duration of 25 minutes. Because the validity interval is 10 minutes,
    # the agent is only asked to get an action twice more during this time (for
    # a total of three actions).
    self.assertEqual(mock_step.call_count, 5)
    self.assertEqual(mock_get_action_context.call_count, 3)


if __name__ == '__main__':
  absltest.main()
