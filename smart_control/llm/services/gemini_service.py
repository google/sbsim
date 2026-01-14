# pylint: disable=line-too-long
r"""A Gemini service that uses the Gemini API directly, using an API key.

Run with blaze:

```shell
$ blaze run //third_party/py/smart_buildings/smart_control/llm/services:gemini_service_script
```

Run with python:

```shell
$ python -m smart_buildings.smart_control.llm.services.gemini_service
```

Optional flags:
  --gemini_api_key: API key to use for the Gemini API.
  --gemini_model_temperature: The model temperature.

Example:

```shell
$ blaze run //third_party/py/smart_buildings/smart_control/llm/services:gemini_service_script -- \
  --gemini_api_key=<api_key> --gemini_model_temperature=0.5
```
"""
# pylint: enable=line-too-long

import abc
import getpass
import os
from typing import Any, Sequence

from absl import app
from absl import flags
import dotenv
from google import genai
from smart_buildings.smart_control.llm.services import llm_service


dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', default='gemini-2.0-flash')

TEMPERATURE = 0.1
TOP_P = 0.95
TOP_K = 40
MAX_OUTPUT_TOKENS = 1024


FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='gemini_api_key',
    default=None,
    help='API key to use for the Gemini API.',
)

flags.DEFINE_string(
    name='gemini_model_temperature', default=None, help='The model temperature.'
)


class BaseGeminiService(llm_service.BaseLLMService, metaclass=abc.ABCMeta):
  """A Gemini service interface allowing for flexible credentials approaches.

  Attributes:
    model_name: The name of the Gemini model to use.
    temperature: Controls the randomness of the output. Higher values mean more
      random, lower values mean more deterministic.
    top_p: Nucleus sampling parameter. Considers the smallest set of tokens
      whose cumulative probability exceeds this value.
    top_k: Top-k sampling parameter. Considers the top k most likely tokens at
      each step.
    max_output_tokens: The maximum number of tokens to generate.
    generation_config: The generation config to use for the model.
    api_key: The API key to use for the Gemini API.
    client: The model client.
  """

  def __init__(
      self,
      model_name: str = MODEL_NAME,
      temperature: float = TEMPERATURE,
      top_p: float = TOP_P,
      top_k: float = TOP_K,
      max_output_tokens: int = MAX_OUTPUT_TOKENS,
  ):
    """Initializes a Gemini service interface.

    Args:
      model_name: The name of the Gemini model to use.
      temperature: Controls the randomness of the output. Higher values mean
        more random, lower values mean more deterministic.
      top_p: Nucleus sampling parameter. Considers the smallest set of tokens
        whose cumulative probability exceeds this value.
      top_k: Top-k sampling parameter. Considers the top k most likely tokens at
        each step.
      max_output_tokens: The maximum number of tokens to generate.
    """
    self._model_name = model_name
    self._temperature = temperature
    self.top_p = top_p
    self.top_k = top_k
    self._max_output_tokens = max_output_tokens

  @property
  def json_metadata(self) -> dict[str, Any]:
    """Info to write into a JSON file. Needs to be serializable."""
    return {
        'type': self.__class__.__name__,
        'model_name': self.model_name,
        'generation_config': self.generation_config,
    }

  @property
  def model_name(self) -> str:
    return self._model_name

  @property
  def temperature(self) -> float:
    return self._temperature

  @property
  def max_output_tokens(self) -> int:
    return self._max_output_tokens

  @property
  @abc.abstractmethod
  def client(self) -> genai.Client:
    """Returns a client for the Gemini service."""

  @property
  def generation_config(self) -> dict[str, Any]:
    return {
        'temperature': self.temperature,
        'top_p': self.top_p,
        'top_k': self.top_k,
        'max_output_tokens': self.max_output_tokens,
    }

  def get_response(self, prompt: str) -> str:
    """Returns the response from the Gemini model."""
    response = self.client.models.generate_content(
        model=self.model_name, contents=prompt, config=self.generation_config
    )
    return response.text


class GeminiService(BaseGeminiService):
  """A Gemini service that uses the Gemini API directly, using an API key.

  Will use the `GEMINI_API_KEY` environment variable if provided.
  """

  def __init__(
      self,
      api_key: str = GEMINI_API_KEY,
      model_name: str = MODEL_NAME,
      temperature: float = TEMPERATURE,
      top_p: float = TOP_P,
      top_k: float = TOP_K,
      max_output_tokens: int = MAX_OUTPUT_TOKENS,
      client: genai.Client | None = None,
  ):
    """Initializes the Gemini service.

    Args:
      api_key: The API key for the Gemini API. Will use the `GEMINI_API_KEY`
        environment variable if provided.
      model_name: The name of the Gemini model to use.
      temperature: Controls the randomness of the output. Higher values mean
        more random, lower values mean more deterministic.
      top_p: Nucleus sampling parameter. Considers the smallest set of tokens
        whose cumulative probability exceeds this value.
      top_k: Top-k sampling parameter. Considers the top k most likely tokens at
        each step.
      max_output_tokens: The maximum number of tokens to generate.
      client: An optional client to use for the Gemini API. Primarily used to
        facilitate dependency injection during testing. If not provided, a new
        client will be created using the specified api_key.
    """
    super().__init__(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
    )

    if not api_key:
      raise ValueError(
          'Please provide an api_key, or set the GEMINI_API_KEY '
          'environment variable.'
      )
    self.api_key = api_key

    self._client = client or genai.Client(api_key=self.api_key)

  @property
  def client(self) -> genai.Client:
    return self._client


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  api_key = FLAGS.gemini_api_key or getpass.getpass('API Key: ') or GEMINI_API_KEY  # pylint: disable=line-too-long
  temp = FLAGS.gemini_model_temperature or input('Temperature: ') or TEMPERATURE
  service = GeminiService(api_key=api_key, temperature=temp)

  user_prompt = input('Prompt: ') or 'When was America founded?'
  print(service.get_response(user_prompt))


if __name__ == '__main__':
  app.run(main)
