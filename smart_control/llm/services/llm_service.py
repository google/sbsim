"""Provides a generic interface for an LLM service."""

import abc
from typing import Any


class BaseLLMService(metaclass=abc.ABCMeta):
  """Base class defining the common interface for an LLM service."""

  @property
  def json_metadata(self) -> dict[str, Any]:
    """Info to write into a JSON file. Needs to be serializable."""
    return {
        "type": self.__class__.__name__,
        "model_name": self.model_name,
        "temperature": self.temperature,
    }

  @property
  @abc.abstractmethod
  def model_name(self) -> str:
    """Returns the LLM model name."""

  @property
  @abc.abstractmethod
  def temperature(self) -> float:
    """Returns the LLM temperature."""

  @abc.abstractmethod
  def get_response(self, prompt: str) -> str | None:
    """Returns the LLM's textual response from a given prompt."""
