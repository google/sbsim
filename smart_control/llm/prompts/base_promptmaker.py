"""Base class for promptmakers.

Promptmakers are responsible for compiling a prompt for an LLM.

Promptmakers are set up to combine a user-provided 'base prompt' with output
formatting instructions automatically derived from a Pydantic model, to arrive
at the final 'prompt' which gets sent to the LLM.

This base class can be flexibility used with any Pydantic model, but child
classes will use a specific Pydantic model suited for building control.
"""

import abc
import textwrap
from typing import Any, Callable

import langchain.output_parsers
import pydantic

PydanticOutputParser = langchain.output_parsers.PydanticOutputParser

SerializableData = dict[str, Any]

DedentFunction = Callable[[str], str]


def full_dedent(txt: str) -> str:
  """Removes all leading whitespace from each line in a string.

  While textwrap.dedent is designed to preserve the relative indentation within
  a block of text, this function removes all leading whitespace from each line,
  regardless of the relative indentation.

  This behavior is helpful when you want to define a prompt as a multiline
  string inside a function or method, and you want to ensure all lines in the
  resulting prompt are left-justified, ignoring any indentation used for
  readability in the source code.

  This is also relevant when a prompt is dynamically compiled using
  multiple sections, including nested sub-sections that are defined in their own
  methods in the promptmaker class. This behavior can prevent stacking of
  relative indentation from nested blocks of code.

  If you have a markdown multi-level list, you would want to use
  textwrap.dedent instead, to preserve the relative indentation of the list.

  Args:
    txt: The string to remove leading whitespace from.

  Returns:
    The string with all leading whitespace removed.
  """
  return '\n'.join(line.lstrip() for line in txt.strip().splitlines())


class BasePromptmaker(abc.ABC):
  """Base Promptmaker.

  A Promptmaker is responsible for compiling a prompt for an LLM.

  The Promptmaker uses a Pydantic model to provide formatting instructions that
  result in the LLM producing reliable JSON formatted string responses.

  You override the `base_prompt` property to provide the main prompt, and the
  `output_schema_class` argument to specify the Pydantic model used to provide
  formatting instructions. Then the promptmaker combines your base prompt with
  formatting instructions in the final `prompt` property, which you can send to
  the LLM.
  """

  def __init__(
      self,
      output_schema_class: type[pydantic.BaseModel],
      dedent: DedentFunction = textwrap.dedent,
  ):
    """Initializes the instance.

    Args:
      output_schema_class: The pydantic model class used to provide JSON
        response formatting instructions in the prompt.
      dedent: The function used to remove leading whitespace from the prompt.
    """
    self.output_schema_class = output_schema_class
    self.dedent = dedent

  @property
  @abc.abstractmethod
  def base_prompt(self) -> str:
    """The main prompt, fully hydrated with data as necessary.

    The `base_prompt` does not include formatting instructions, as they are
    automatically added in the `prompt` property.
    """

  @property
  def prompt(self) -> str:
    """The final prompt, including response formatting instructions."""
    return self.dedent(
        '\n\n'.join((
            self.base_prompt,
            self.formatting_instructions_section,
        ))
    )

  @property
  def formatting_instructions_section(self) -> str:
    """The section of the prompt containing formatting instructions."""
    return '\n'.join([
        '## Formatting Instructions\n',
        (
            'IMPORTANT: The output MUST be a single, valid JSON object '
            'conforming to the schema below.'
        ),
        (
            'Do NOT include any other text, explanations, pleasantries, or any '
            'other content before or after the JSON object.'
        ),
        self.formatting_instructions,
    ])

  @property
  def formatting_instructions(self) -> str:
    """Formatting instructions for the desired LLM output structure."""
    return self.output_parser.get_format_instructions()

  @property
  def output_parser(self) -> PydanticOutputParser:
    """A parser that derives formatting instructions from a pydantic model."""
    return PydanticOutputParser(pydantic_object=self.output_schema_class)

  @property
  def output_schema(self) -> dict[str, Any]:
    """The JSON schema for the output."""
    return self.output_schema_class.model_json_schema()

  @property
  def json_metadata(self) -> SerializableData:
    """Metadata about the promptmaker, suitable for JSON serialization."""
    return {
        'type': self.__class__.__name__,
        'output_schema_class': self.output_schema_class.__name__,
    }
