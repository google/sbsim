"""LLM agent.

This agent uses a large language model (LLM) to determine its actions. The agent
can use any LLM service that implements the `BaseLLMService` interface.

First, the agent gets current building conditions from the environment, and
passes this information to a promptmaker, which is responsible for dynamically
constructing a prompt.

The agent then passes the prompt to the LLM using the configured LLM service.
The agent then validates the LLM's response to ensure it is JSON-formatted and
adheres to the specified "action" output schema. If the action is valid, the
agent stores this information for future reference, and returns the action to
the control loop.

If the LLM isn't able to produce a valid response, the agent keeps a record of
the error(s), and tries again (with exponential backoff), until it receives a
valid response or reaches the maximum number of tries.

If the agent doesn't receive a valid action after trying the maximum number of
times, it logs a record of this max retries exceeded error, and gracefully uses
a fallback action:

  + If a previous valid action is available, the agent uses a modified version
    of its most recent action, except it uses the shortest possible validity
    interval, to give the agent a chance to get a new action at the next
    available opportunity.
  + If a previous action isn't available, the agent falls back to using the
    environment's normally scheduled default action.

Since this agent inherits from the Schedule Policy Agent to determine the
normally scheduled action, it should be used in conjunction with a hybrid action
environment.
"""

import re
from typing import Any, override

from absl import logging
import backoff
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import base_agent
from smart_buildings.smart_control.llm.agents import schedule_agent
from smart_buildings.smart_control.llm.prompts import promptmaker as pm
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.services import llm_service as llm
from smart_buildings.smart_control.llm.utils import schedule_tool as schedule_lib
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import serialization
from smart_buildings.smart_control.utils import temperature_conversion as temp

_MARKDOWN_CODE_BLOCK_RE = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)


def parse_response_text(txt: str | None) -> str:
  """Parses and cleans the raw text response from the LLM.

  The response text is expected to be a JSON-formatted string. In practice
  we often see the response wrapped in a JSON-formatted markdown code block,
  even when we instruct the LLM to not do that. Perhaps it sees the markdown
  formatting in the prompt and thinks it should use markdown formatting in the
  response as well. So this method will try to strip the markdown code block
  formatting to ensure the text is valid JSON.

  Args:
    txt: The raw response text returned by the LLM.

  Returns:
    The LLM's textual response as a valid JSON-formatted string.

  Raises:
    ValueError: If the response text is not a string.
  """
  # FYI: When using the Gemini API, sometimes the response text is None.
  # For example, in the case of a max tokens error.
  if not isinstance(txt, str):
    raise ValueError('Expecting a string response')

  # If the response is wrapped in a markdown code block, extract it:
  match = _MARKDOWN_CODE_BLOCK_RE.search(txt)
  if match:
    return match.group(1).strip()

  # Otherwise, fallback to stripping standard markdown fences and whitespace:
  return txt.replace('```json', '').replace('```', '').strip()


class MaxRetriesExceededError(Exception):
  """Maximum number of retries met or exceeded."""

  pass


class LLMAgent(schedule_agent.SchedulePolicyAgent):
  """LLM agent.

  Attributes:
    llm_service: The LLM service to use for generating responses.
    output_schema_class: The Pydantic model class used to validate and parse the
      LLM's JSON response.
    temp_display_unit: The temperature unit used for displaying temperatures in
      the LLM's response justifications.
    promptmaker: The promptmaker instance used to construct the LLM prompt.
    max_tries: The maximum number of times to attempt calling the LLM if parsing
      or validation errors occur. After this limit is reached, the agent will
      fallback to using the scheduled action context.
  """

  def __init__(
      self,
      *,
      env: hybrid_action_environment.HybridActionEnvironment,
      llm_service: llm.BaseLLMService,
      promptmaker: pm.Promptmaker | None = None,
      promptmaker_class: type[pm.Promptmaker] | None = None,
      max_tries: int = 5,
      clip: bool = True,
      override_discrete_defaults: bool = True,
      schedule_tool: schedule_lib.BuildingScheduleTool | None = None,
  ):
    """Initializes the instance.

    Pass either a promptmaker instance or a promptmaker class. The promptmaker
    class will only be used if a promptmaker instance is not provided. The
    promptmaker class is a convenience argument because we usually want to use
    the same promptmaker arguments, and will just change the promptmaker class
    to represent different buildings or custom validity intervals.

    Args:
      env: The environment in which the agent will operate.
      llm_service: The LLM service to use for generating responses.
      promptmaker: The promptmaker instance used to construct the LLM prompt.
      promptmaker_class: The promptmaker class used to construct the LLM prompt.
        If a promptmaker instance is provided, this argument will be ignored,
        otherwise a promptmaker instance will be created using this class, using
        reasonable default arguments.
      max_tries: The maximum number of times to attempt calling the LLM if
        parsing or validation errors occur.
      clip: Whether to clip the generated setpoints to be within the valid
        ranges defined by the environment.
      override_discrete_defaults: Whether to override discrete defaults when
        getting the scheduled action context.
      schedule_tool: Optionally provide a BuildingScheduleTool instance.
        Otherwise, a schedule tool will be constructed using the agent's
        environment and default schedule tool arguments.
    """
    super().__init__(
        env=env,
        schedule_tool=schedule_tool,
        clip=clip,
        override_discrete_defaults=override_discrete_defaults,
    )

    self.llm_service = llm_service
    self.promptmaker = self._setup_promptmaker(promptmaker, promptmaker_class)
    self.output_schema_class = self.promptmaker.output_schema_class
    self.temp_display_unit = self.promptmaker.temp_display_unit
    self.max_tries = max_tries

    self._last_attempt_response_text: str | None = None
    self._last_valid_llm_action: action_context.ActionContext | None = None

    # Wrap `_attempt_get_action_context` to retry with exponential backoff.
    self._retry_attempt_get_action_context_with_backoff = backoff.on_exception(
        wait_gen=backoff.expo,
        exception=Exception,
        max_tries=self.max_tries,
        jitter=backoff.full_jitter,
        on_backoff=self._on_backoff,
        on_giveup=self._on_giveup,
    )(
        self._attempt_get_action_context  # the method being wrapped
    )

  def _setup_promptmaker(
      self,
      promptmaker: pm.Promptmaker | None = None,
      promptmaker_class: type[pm.Promptmaker] | None = None,
  ) -> pm.Promptmaker:
    """Sets up the promptmaker instance."""
    if (promptmaker is None and promptmaker_class is None) or (
        promptmaker is not None and promptmaker_class is not None
    ):
      raise ValueError(
          'Either a promptmaker instance or class must be provided, not both.'
      )

    return promptmaker or promptmaker_class(
        env=self.env,
        observation_response=None,
        reward_info=None,
        lazy_init_protos=True,
        output_schema_class=action_context.HybridActionContext,
        temp_display_unit=temp.TempUnit.FAHRENHEIT,
        include_weights=True,
    )

  @override
  @property
  def json_metadata(self) -> serialization.SerializableData:
    return super().json_metadata | {
        'llm_service': self.llm_service.json_metadata,
        'promptmaker': self.promptmaker.json_metadata,
        'output_schema': {'type': self.output_schema_class.__name__},
        'temp_display_unit': self.temp_display_unit.value,
        'max_tries': self.max_tries,
    }

  # PROMPT

  def make_prompt(
      self,
      observation_response: building_pb2.ObservationResponse,
      reward_info: reward_pb2.RewardInfo,
  ) -> str:
    """Creates a prompt, using the provided promptmaker class.

    Args:
      observation_response: The observation response from the environment.
      reward_info: The reward info from the environment.

    Returns:
      The prompt to be sent to the LLM.
    """
    self.promptmaker.set_protos(
        observation_response=observation_response, reward_info=reward_info
    )
    return self.promptmaker.prompt

  # RESPONSE VALIDATION

  def validate_action_context(self, txt: str) -> action_context.ActionContext:
    """Ensures the response text is in the expected JSON format.

    Args:
      txt: The response text to validate.

    Raises:
      pydantic.ValidationError: If the response text is not valid JSON.

    Returns:
      The validated action context object.
    """
    if issubclass(self.output_schema_class, action_context.ActionContext):
      return self.output_schema_class.from_json(
          txt=txt, env=self.env, clip=self._clip
      )

    action = self.output_schema_class.model_validate_json(txt)
    return self.action_context_class(
        env=self.env, clip=self._clip, **action.model_dump()
    )

  # ACTION

  def _attempt_get_action_context(
      self,
      prompt: str,
  ) -> action_context.ActionContext:
    """Attempts to get a valid action from the LLM.

    Clears and resets the last response text that has been received from the
    LLM.

    FYI: When using the Gemini API, sometimes the response text is None.
    For example, in the case of a max tokens error.

    Args:
      prompt: The prompt to send to the LLM service.

    Raises:
      ValueError: If the LLM service returns None.
      JSONDecodeError: If the response text is string but not valid JSON.
      pydantic.ValidationError: If the JSON doesn't adhere to the output schema.

    Returns:
      A validated action context object.
    """
    self._last_attempt_response_text = None
    response_text = self.llm_service.get_response(prompt)
    self._last_attempt_response_text = response_text

    if response_text is None:
      raise ValueError('LLM service returned None')

    action = self.validate_action_context(parse_response_text(response_text))
    self._last_valid_llm_action = action
    return action

  def _record_backoff_error(
      self,
      error_details: dict[str, Any],
  ) -> None:
    """Consolidates logic for recording error details returned by backoff.

    When using the @backoff.on_exception decorator, the details dictionary
    passed to the on_backoff and on_giveup handler functions contains the
    following keys:

    - target: The decorated function that is being retried.
    - args: The positional arguments passed to the target function.
    - kwargs: The keyword arguments passed to the target function.
    - tries: The number of attempts made so far.
    - elapsed: The time in seconds elapsed since the first attempt.
    - exception: The exception instance that was caught and triggered the
      backoff or giveup.

    Specifically for the on_backoff handler, the details dictionary also
    includes:

    - wait: The calculated number of seconds to wait before the next retry
      attempt.

    Args:
      error_details: The error details returned by backoff.
    """
    exception = error_details['exception']
    nested_errors = exception.errors() if hasattr(exception, 'errors') else None

    self.errors.append(
        base_agent.AgentErrorRecord(
            error_type=exception.__class__.__name__,
            error_message=repr(exception),
            details=nested_errors,
            metadata={
                'tries': error_details.get('tries'),
                'elapsed': error_details.get('elapsed'),
                'wait': error_details.get('wait'),
                'response_text': self._last_attempt_response_text,
            },
        )
    )

  def _on_backoff(self, details: dict[str, Any]) -> None:
    """Records an error that occurred during a backoff retry."""
    logging.debug('ON BACKOFF: %r', details)
    self._record_backoff_error(details)

  def _on_giveup(self, details: dict[str, Any]) -> None:
    """Records final error and exhaustion of retries."""

    # Record the final specific error that caused the giveup.
    logging.debug('ON GIVEUP: %r', details)
    self._record_backoff_error(details)

    # Record that max retries were exceeded.
    exhaustion_error = base_agent.AgentErrorRecord(
        error_type=MaxRetriesExceededError.__name__,
        error_message=f'Max tries ({self.max_tries}) exceeded.',
        metadata={},
    )
    self.errors.append(exhaustion_error)

  def get_action_context(
      self,
      observation_response: building_pb2.ObservationResponse | None = None,
      reward_info: reward_pb2.RewardInfo | None = None,
  ) -> action_context.ActionContext:
    """Returns the action context to be used within the agent control loop.

    Args:
      observation_response: The observation response from the environment.
      reward_info: The reward info from the environment.

    Returns:
      The action context to be used within the agent control loop.
    """
    prompt = self.make_prompt(observation_response, reward_info)
    try:
      return self._retry_attempt_get_action_context_with_backoff(prompt)
    except Exception:  # pylint: disable=broad-except
      # All retry attempts failed, and on_giveup has recorded the errors.
      if self._last_valid_llm_action is not None:
        logging.exception(
            'LLM MAX TRIES EXCEEDED. FALLING BACK TO PREVIOUS LLM ACTION...',
        )
        return self._last_valid_llm_action.model_copy(
            update={
                'validity_interval': self.env.time_step_mins,
                'justification': 'Previous LLM action (max retries exceeded)',
            }
        )

      logging.exception(
          'LLM MAX TRIES EXCEEDED. NO PREVIOUS LLM ACTION AVAILABLE. FALLING'
          ' BACK TO SCHEDULED ACTION...',
      )
      return self.get_scheduled_action_context()
