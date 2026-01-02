"""Helpers for testing LLM services.

The tests will implement mocked responses by default.

To test the actual responses returned by the Gemini API, optionally set the
`TEST_GEMINI_SERVICE_LIVE` environment variable to 'true'.

To test the actual responses returned by the Vertex AI API, optionally set the
`TEST_VERTEX_SERVICE_LIVE` environment variable to 'true'.
"""

import os
from unittest import mock

import dotenv
from google import genai
from google.auth import credentials
from smart_buildings.smart_control.llm.services import gemini_service
from smart_buildings.smart_control.llm.services import llm_service
from smart_buildings.smart_control.llm.services import vertex_service


dotenv.load_dotenv()

TEST_GEMINI_SERVICE_LIVE = bool(
    os.getenv("TEST_GEMINI_SERVICE_LIVE", default="false").lower() == "true"
)
TEST_VERTEX_SERVICE_LIVE = bool(
    os.getenv("TEST_VERTEX_SERVICE_LIVE", default="false").lower() == "true"
)

SKIP_REASON = "Skip API Calls in tests by default."

PROMPT_TEXT = "What year was America founded?"
RESPONSE_TEXT = (
    "The United States was founded in 1776 after the Declaration of "
    "Independence."
)


class FakeLLMService(llm_service.BaseLLMService):
  """Generic Fake LLM Service, used for testing."""

  def __init__(self, response_text: str = RESPONSE_TEXT):
    self._temperature = 0.0
    self._response_text = response_text

  @property
  def model_name(self) -> str:
    return "fake-model"

  @property
  def temperature(self) -> float:
    return self._temperature

  def get_response(self, prompt: str) -> str:
    return self._response_text


def create_fake_llm_service(
    response_text: str = RESPONSE_TEXT,
) -> FakeLLMService:
  """Creates a fake version of a generic LLM Service.

  It will return the specified response text instead of making an API call.

  Args:
    response_text: The text to return from the LLM Service.

  Returns:
    A fake version of the LLM Service.
  """
  return FakeLLMService(response_text=response_text)


def create_mock_gemini_service(
    response_text: str = RESPONSE_TEXT,
) -> gemini_service.GeminiService:
  """Creates a mock version of the Gemini Service.

  It will return the specified response text instead of making an API call.

  Args:
    response_text: The text to return from the Gemini Service. If not provided,
      a default response text will be used.

  Returns:
    A mock version of the Gemini Service.
  """
  # mocked dependencies:
  client = mock.create_autospec(genai.Client, instance=True)
  generate_content_response = mock.MagicMock()
  generate_content_response.text = response_text
  client.models.generate_content.return_value = generate_content_response

  # dependency injection:
  return gemini_service.GeminiService(api_key="fake_api_key", client=client)


def create_mock_vertex_service(
    response_text: str = RESPONSE_TEXT,  # pylint: disable=unused-argument
) -> vertex_service.VertexAIService:
  """Creates a mock version of the Vertex AI Service.

  It will return the specified response text instead of making an API call.

  Args:
    response_text: The text to return from the Vertex AI Service. If not
      provided, a default response text will be used.

  Returns:
    A mock version of the Vertex AI Service.
  """
  # mocked credentials:
  creds = mock.create_autospec(credentials.Credentials, instance=True)

  # mocked client:
  client = mock.create_autospec(genai.Client, instance=True)
  generate_content_response = mock.MagicMock()
  generate_content_response.text = response_text
  client.models.generate_content.return_value = generate_content_response

  # dependency injection:
  return vertex_service.VertexAIService(
      project_id="not-a-real-project", credentials=creds, client=client
  )
