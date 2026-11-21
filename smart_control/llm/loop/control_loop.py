"""Agent control loop.

The loop is a harness / driver to facilitate an agent's control of an
environment. The loop is responsible for getting observations from the
environment, getting actions from the agent, and stepping the environment to
apply those actions to the building.

The loop runs a single episode, covering a specified number of days according to
the environment's configuration. It steps the environment on a regular time step
interval (usually every five minutes), as specified by the environment's
configuration.

**Validity Interval**

Some agents (like RL agents and baseline agents) may take actions
every time step, while others (like LLM agents) may choose to specify longer
validity intervals. The validity interval is the amount of time for which an
action is valid (i.e. the amount of time to wait before asking the agent for
another action). While the loop is waiting for the validity interval to expire,
it will apply the most recent action it has received, to step the environment
during every time step until the validity interval runs out.

The loop will step the environment at every time step, but will only ask the
agent for a new action when the validity interval runs out. Agents like
baseline agents or RL agents that don't vary their validity intervals can use
the environment's time step interval in minutes, as a fixed default interval for
every action. Other agents like LLM agents may choose to specify longer validity
intervals, based on building conditions - for example an agent may choose to
wait two hours between actions, at night when conditions are stable and there
are no occupants in the building.

The validity interval also acts as a cost-saving measure, as it can reduce the
number of API calls to the LLM (from around 288 to around 25 per day).

**Action Context**

The agent provides an action context object to the loop, which the loop uses to
step the environment. The action context contains the action itself, as well as
more context about the action, suchj as the validity interval, and
justifications / reasoning, as applicable.

**Max Steps**

The loop can be stopped early if a maximum number of steps is specified. This is
helpful for testing and debugging purposes.

**Metrics**

The basic control loop uses existing metrics writing functionality, triggering
protos to be written to file during each time step (see environment's methods to
get information about observations and rewards).
"""

import logging
from typing import Any, Final

import numpy as np
import pandas as pd
from smart_buildings.smart_control.llm.agents import default_agent
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import writer_lib
from tf_agents.trajectories import time_step as ts

SerializableData = dict[str, Any]

ACTION_REJECTION_REWARD: Final[float] = -np.inf


def get_clock_timestamp() -> pd.Timestamp:
  """Returns the actual current clock timestamp."""
  return pd.Timestamp.now().replace(microsecond=0, nanosecond=0)


def parse_timestamp(timestamp: pd.Timestamp, time_zone: str) -> pd.Timestamp:
  """Ensures that a timestamp is timezone-aware."""
  if timestamp.tzinfo is None:
    return timestamp.tz_localize(time_zone)
  return timestamp.tz_convert(time_zone)


class ControlLoop:
  """An agentic control loop.

  The loop is responsible for stepping the environment on a regular basis.

  The agent is called to get an action whenever the validity interval runs out.

  If a maximum number of steps is specified, the loop will stop running after
  that number of steps.

  The loop will keep track of the agent's cumulative rewards over time.

  Attributes:
    agent: The agent to use for the loop.
    env: The environment to use for the loop.
    metrics_writer: The metrics writer to use for the loop.
    max_steps: The maximum number of steps to run the loop for.
    cum_reward: The cumulative reward for the loop.
    results: The results of the loop.
  """

  def __init__(
      self,
      agent: default_agent.DefaultPolicyAgent,
      max_steps: int | None = None,
  ):
    """Initializes the instance.

    Args:
      agent: The agent to use for the loop.
      max_steps: The maximum number of steps to run the loop for. If None, the
        loop will run until the environment has ended.
    """
    self.agent = agent
    self.env = self.agent.env
    self.metrics_writer = self._validate_metrics_writer(self.env.metrics_writer)

    self.max_steps = max_steps

    self.cum_reward = 0.0
    self.results = []

  def _interval_has_expired(self, remaining_interval: pd.Timedelta) -> bool:
    """Checks whether the validity interval has expired.

    If so, it is time to get a new action from the agent.

    Args:
      remaining_interval: timedelta representing the remaining interval to wait
        before getting a new action from the agent.

    Returns:
      Whether or not the interval has expired.
    """
    return remaining_interval <= self.time_step_interval

  def _max_steps_reached(self, max_step: int | None) -> bool:
    return max_step is not None and self.current_step >= max_step

  def _action_rejected(self, time_step: ts.TimeStep) -> bool:
    """Checks whether the action was rejected by the environment."""
    return (time_step.reward == ACTION_REJECTION_REWARD).any()

  #
  # MAIN LOOP
  #

  def run(self) -> None:
    """Runs the control loop for a single episode."""
    self.write_metadata()

    max_step = (
        self.current_step + self.max_steps
        if self.max_steps is not None
        else None
    )

    # GET INITIAL STATE

    observation_response = self.env.get_observation_response()
    reward_info, reward_response = self.env.get_reward_info_and_response()

    # GET INITIAL AGENT ACTION

    action_ctx = self.agent.get_action_context(
        observation_response=observation_response,
        reward_info=reward_info,
    )
    action = action_ctx.get_action()
    remaining_interval = pd.Timedelta(minutes=action_ctx.validity_interval)

    while True:
      if self.episode_has_ended:
        logging.info("EPISODE HAS ENDED. STOPPING...")
        break

      if self._max_steps_reached(max_step):
        logging.info("MAX STEPS REACHED. STOPPING...")
        break

      # STEP THE ENV (USING WHATEVER ACTION IT HAS MOST RECENTLY RECEIVED)

      time_step = self.env.step(action)
      if self._action_rejected(time_step):
        logging.warning("ACTION REJECTED BY THE ENVIRONMENT.")

      reward = time_step.reward.item()
      self.cum_reward += float(reward)
      logging.info("REWARD: %r --> %r", reward, self.cum_reward)

      # UPDATE RESULTS

      self.update_results(
          reward=reward,
          reward_info=reward_info,
          reward_response=reward_response,
      )

      # GET NEW STATE

      observation_response = self.env.get_observation_response()
      reward_info, reward_response = self.env.get_reward_info_and_response()

      # UPDATE ACTION (AS NECESSARY)

      if self._interval_has_expired(remaining_interval):
        # VALIDITY INTERVAL HAS EXPIRED. GET A NEW ACTION FROM THE AGENT.
        action_ctx = self.agent.get_action_context(
            observation_response=observation_response,
            reward_info=reward_info,
        )
        action = action_ctx.get_action()
        remaining_interval = pd.Timedelta(minutes=action_ctx.validity_interval)
      else:
        # CONTINUE WAITING FOR VALIDITY INTERVAL TO EXPIRE
        remaining_interval -= self.time_step_interval

    # EPISODE HAS ENDED

    self.write_results()

  #
  # ENVIRONMENT PROPERTIES
  #

  @property
  def start_timestamp(self) -> pd.Timestamp:
    """The start timestamp, in environment's local time zone."""
    return parse_timestamp(self.env.start_timestamp, self.env.time_zone)

  @property
  def end_timestamp(self) -> pd.Timestamp:
    """The end timestamp, in the environment's local time zone."""
    return parse_timestamp(self.env.end_timestamp, self.env.time_zone)

  @property
  def days_per_episode(self) -> int:
    """The number of steps per episode."""
    return self.env.num_days_in_episode

  @property
  def time_step_interval(self) -> pd.Timedelta:
    """The time step in minutes, as a pandas Timedelta."""
    return pd.Timedelta(minutes=self.env.time_step_mins)

  @property
  def steps_per_day(self) -> int:
    """The number of steps per day."""
    return int(pd.Timedelta(days=1) / self.time_step_interval)

  @property
  def steps_per_episode(self) -> int:
    """The number of steps per episode."""
    return self.env._num_timesteps_in_episode  # pylint: disable=protected-access

  @property
  def episode_has_ended(self) -> bool:
    """Whether the episode has ended."""
    return self.env._has_episode_ended()  # pylint: disable=protected-access

  @property
  def current_step(self) -> int:
    """The current step number."""
    return self.env._step_count  # pylint: disable=protected-access

  @property
  def current_local_timestamp(self) -> pd.Timestamp:
    """The current local timestamp."""
    return self.env.current_local_timestamp

  #
  # METRICS
  #

  def _validate_metrics_writer(
      self, writer: writer_lib.BaseWriter
  ) -> writer_lib.BaseWriter:
    """Validates the metrics writer."""
    if writer is None:
      raise ValueError("Metrics writer is None.")

    if not hasattr(writer, "output_dir"):
      raise ValueError("Metrics writer does not have output_dir attribute.")

    if not hasattr(writer, "write_json"):
      raise ValueError("Metrics writer does not have write_json method.")

    return writer

  @property
  def metrics_output_dir(self) -> Any:
    """The directory to write metrics to."""
    return self.metrics_writer.output_dir

  def write_metadata(self) -> None:
    """Writes the metadata to a file (before running the loop)."""
    self.metrics_writer.write_json(self.json_metadata, "metadata.json")

  @property
  def json_metadata(self) -> SerializableData:
    """Info about the loop's initial state and input parameters."""
    return {
        "start_timestamp": str(self.start_timestamp),
        "end_timestamp": str(self.end_timestamp),
        "days_per_episode": self.days_per_episode,
        "time_step_mins": self.env.time_step_mins,
        "steps_per_episode": self.steps_per_episode,
        "env": self.env.json_metadata,
        "agent": self.agent.json_metadata,
    }

  def update_results(
      self,
      reward: float,
      reward_info: reward_pb2.RewardInfo,
      reward_response: reward_pb2.RewardResponse,
  ) -> None:
    """Updates the results (after the current step has completed)."""
    pass

  def write_results(self) -> None:
    """Writes the results to a file (after the episode has completed)."""
    self.metrics_writer.write_json(self.json_results, "results.json")

  @property
  def json_results(self) -> SerializableData:
    """Info about the loop's current / final state, after it has begun."""
    return {
        "clock_timestamp": str(get_clock_timestamp()),
        "current_timestamp": str(self.current_local_timestamp),
        "current_step": self.current_step,
        "cum_reward": self.cum_reward,
        "results": self.results,
    }
