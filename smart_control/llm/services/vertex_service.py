"""A Gemini Service that uses the Vertex AI platform, and a GCP project."""

import os
from typing import Any

import dotenv
from google import auth
from google import genai
from google.genai import types
from smart_buildings.smart_control.llm.services import gemini_service


dotenv.load_dotenv()

CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", default="smart-buildings-dev")
LOCATION = os.getenv("VERTEX_AI_LOCATION", default="us-central1")
MODEL_NAME = os.getenv("VERTEX_AI_MODEL_NAME", default="gemini-2.5-flash")

SAFETY_DISABLED = (
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
    ),
)


class VertexAIService(gemini_service.BaseGeminiService):
  """A Gemini Service that uses Vertex AI and a GCP project.

  Attributes:
    project_id: The GCP project ID to use for the Vertex AI service.
    location: The GCP location to use for the Vertex AI service.
    credentials: The credentials to use for the Vertex AI service.
    safety_settings: The safety settings to use for the Vertex AI service.
    client: The client to use for the Vertex AI service.
  """

  def __init__(
      self,
      project_id: str | None = PROJECT_ID,
      location: str = LOCATION,
      model_name: str = MODEL_NAME,
      temperature: float = gemini_service.TEMPERATURE,
      top_p: float = gemini_service.TOP_P,
      top_k: float = gemini_service.TOP_K,
      max_output_tokens: int = gemini_service.MAX_OUTPUT_TOKENS,
      safety_settings: list[types.SafetySetting] | None = None,
      credentials: auth.credentials.Credentials | None = None,
      client: genai.Client | None = None,
  ):
    super().__init__(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
    )

    self.project_id = project_id
    self.location = location
    self.credentials = credentials or CREDENTIALS
    self.safety_settings = safety_settings or SAFETY_DISABLED

    # use default credentials if not provided:
    if not self.credentials and not self.project_id:
      self.credentials, self.project_id = auth.default()

    self._client = client or genai.Client(
        vertexai=True,
        project=self.project_id,
        location=self.location,
        credentials=self.credentials,
    )

  @property
  def client(self) -> genai.Client:
    """Returns a client for the Vertex AI service."""
    return self._client

  @property
  def generation_config(self) -> dict[str, Any]:
    """Returns the generation config for the Vertex AI service."""
    config = super().generation_config.copy()
    config["safety_settings"] = self.safety_settings
    return config
