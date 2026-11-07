"""Factories and helpers for control loop tests."""

from unittest import mock

import pandas as pd
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.agents import default_agent
from smart_buildings.smart_control.llm.loop import control_loop
from smart_buildings.smart_control.utils import writer_lib

START_TIMESTAMP = pd.Timestamp('2025-12-12 00:00:00', tz='US/Pacific')


def create_loop(
    start_timestamp: pd.Timestamp = START_TIMESTAMP,
    loop_class: type[control_loop.ControlLoop] = control_loop.ControlLoop,
    max_steps: int | None = 5,
    hybrid: bool = True,
    agent: default_agent.DefaultPolicyAgent | None = None,
) -> control_loop.ControlLoop:
  """Creates a control loop, with a default agent, for testing purposes.

  Args:
    start_timestamp: The start timestamp for the environment / building.
    loop_class: The class of the loop to be created.
    max_steps: The maximum number of steps to run the loop for.
    hybrid: Whether to create a hybrid action environment. Default is True.
    agent: The agent to use for the loop. A default agent will be created if
      None.

  Returns:
    A control loop, for testing purposes.
  """

  if hybrid:
    env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT,
        start_timestamp=start_timestamp,
        default_actions=env_conftest.DEFAULT_HYBRID_ACTIONS,
    )
  else:
    env = env_conftest.create_environment(
        layout=env_conftest.DEMO_LAYOUT,
        start_timestamp=start_timestamp,
        default_actions=env_conftest.DEFAULT_ACTIONS,
    )

  env._metrics_writer = mock.create_autospec(  # pylint: disable=protected-access
      writer_lib.BaseWriter, instance=True
  )

  agent = agent or default_agent.DefaultPolicyAgent(env=env, clip=True)

  return loop_class(agent=agent, max_steps=max_steps)
