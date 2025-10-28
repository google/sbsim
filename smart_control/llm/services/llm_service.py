"""Provides a generic interface for an LLM service."""

import abc


class BaseLLMService(metaclass=abc.ABCMeta):
  """Base class defining the common interface for an LLM service."""

  @property
  @abc.abstractmethod
  def temperature(self) -> float:
    """Returns the LLM temperature."""

  @abc.abstractmethod
  def get_response(self, prompt: str) -> str:
    """Returns the LLM's textual response from a given prompt."""
