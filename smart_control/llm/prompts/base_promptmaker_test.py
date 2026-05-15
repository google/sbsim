import json
import textwrap
from typing import Callable

from absl.testing import absltest
import immutabledict
import langchain
import pydantic

from smart_buildings.smart_control.llm.prompts import base_promptmaker
from smart_buildings.smart_control.llm.schema import conftest as schema_conftest


BASE_PROMPT = "What year was America founded?"

EXPECTED_OUTPUT_SCHEMA = immutabledict.immutabledict({
    "title": "ExampleOutputSchema",
    "description": (
        "Simple example implementation of an output schema, for testing"
        " purposes."
    ),
    "type": "object",
    "properties": {
        "year": {
            "description": "The year, as an integer.",
            "title": "Year",
            "type": "integer",
        },
        "explanation": {
            "description": (
                "The reasoning behind choosing this specific year."
            ),
            "title": "Explanation",
            "type": "string",
        },
    },
    "required": ["year", "explanation"],
})


class ExampleOutputSchema(pydantic.BaseModel):
  """Simple example implementation of an output schema, for testing purposes."""

  year: int = pydantic.Field(description="The year, as an integer.")

  explanation: str = pydantic.Field(
      description="The reasoning behind choosing this specific year."
  )


class ExamplePromptmaker(base_promptmaker.BasePromptmaker):
  """Simple example implementation of BasePromptmaker, for testing purposes."""

  def __init__(self, dedent: Callable[[str], str] = textwrap.dedent):
    super().__init__(output_schema_class=ExampleOutputSchema, dedent=dedent)

  @property
  def base_prompt(self) -> str:
    return BASE_PROMPT


#
# TESTS
#


class DedentTest(absltest.TestCase):
  """Tests to contrast different dedentation behavior."""

  def setUp(self):
    super().setUp()
    self.base_prompt = """\
    Hello world!
      Hello world!
    """

  def test_no_dedent_leaves_leading_whitespace(self):
    pm = ExamplePromptmaker(dedent=lambda txt: txt)
    self.assertEqual(
        pm.dedent(self.base_prompt),
        "    Hello world!\n      Hello world!\n    ",
    )

  def test_textwrap_dedent_leaves_leading_relative_whitespace(self):
    pm = ExamplePromptmaker(dedent=textwrap.dedent)
    self.assertEqual(
        pm.dedent(self.base_prompt),
        "Hello world!\n  Hello world!\n",
    )

  def test_full_dedent_removes_all_leading_whitespace(self):
    pm = ExamplePromptmaker(dedent=base_promptmaker.full_dedent)
    self.assertEqual(
        pm.dedent(self.base_prompt),
        "Hello world!\nHello world!",
    )


class BasePromptmakerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.promptmaker = ExamplePromptmaker()

  def test_initialization(self):
    self.assertIsInstance(self.promptmaker, base_promptmaker.BasePromptmaker)
    self.assertEqual(self.promptmaker.output_schema_class, ExampleOutputSchema)

  def test_base_prompt(self):
    self.assertEqual(self.promptmaker.base_prompt, BASE_PROMPT)

  def test_prompt(self):
    self.assertEqual(
        self.promptmaker.prompt,
        f"{BASE_PROMPT}\n\n{self.promptmaker.formatting_instructions_section}",
    )

  def test_formatting_instructions_section(self):
    self.assertEqual(
        self.promptmaker.formatting_instructions_section,
        (
            "## Formatting Instructions\n\n"
            "IMPORTANT: The output MUST be a single, valid JSON object "
            "conforming to the schema below.\n"
            "Do NOT include any other text, explanations, pleasantries, or "
            "any other content before or after the JSON object.\n"
            f"{self.promptmaker.formatting_instructions}"
        ),
    )

  def test_formatting_instructions(self):
    instructions = self.promptmaker.formatting_instructions
    self.assertIsInstance(instructions, str)

    parsed_schema = schema_conftest.parse_instructions_schema(instructions)
    expected_schema = dict(EXPECTED_OUTPUT_SCHEMA)  # a shallow copy
    del expected_schema["title"]
    del expected_schema["type"]

    with self.subTest(name="introduces_the_schema"):
      self.assertStartsWith(
          instructions,
          "The output should be formatted as a JSON instance that conforms"
          " to the JSON schema below.",
      )

    with self.subTest(name="provides_an_example_schema"):
      self.assertIn(
          (
              'As an example, for the schema {"properties": {"foo": {"title":'
              ' "Foo", "description": "a list of strings", "type": "array",'
              ' "items": {"type": "string"}}}, "required": ["foo"]}\nthe'
              ' object {"foo": ["bar", "baz"]} is a well-formatted instance of'
              ' the schema. The object {"properties": {"foo": ["bar", "baz"]}}'
              ' is not well-formatted.'
          ),
          instructions,
      )

    with self.subTest(name="provides_output_schema"):
      self.assertEqual(parsed_schema, expected_schema)
      self.assertEndsWith(
          instructions,
          "Here is the output schema:\n```\n"
          + json.dumps(expected_schema)
          + "\n```",
      )

  def test_output_parser(self):
    self.assertIsInstance(
        self.promptmaker.output_parser,
        langchain.output_parsers.PydanticOutputParser,
    )
    self.assertEqual(
        self.promptmaker.output_parser.pydantic_object,
        ExampleOutputSchema,
    )

  def test_output_schema(self):
    self.assertEqual(self.promptmaker.output_schema, EXPECTED_OUTPUT_SCHEMA)

  def test_json_metadata(self):
    self.assertEqual(
        self.promptmaker.json_metadata,
        {
            "type": "ExamplePromptmaker",
            "output_schema_class": "ExampleOutputSchema",
        },
    )


if __name__ == "__main__":
  absltest.main()
