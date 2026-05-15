"""Tests for formatting instructions produced by output schema models.

When prompting the LLM, we use the Langchain output parser to automatically
generate formatting instructions to be included in the prompt. These
instructions are derived from the output schema model, and include a
description of the desired output, as well as the schema itself.
"""

import textwrap

from absl.testing import absltest
import langchain.output_parsers

from smart_buildings.smart_control.llm.schema import conftest
from smart_buildings.smart_control.llm.schema import output_schema

PydanticOutputParser = langchain.output_parsers.PydanticOutputParser


class SchemaParserTest(absltest.TestCase):
  """Tests for the schema parser helper function."""

  def test_parse_instructions_schema(self):
    instructions = textwrap.dedent("""
    ```
    {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        }
      }
    }
    ```
    """)
    schema = conftest.parse_instructions_schema(instructions)
    self.assertDictEqual(
        schema,
        {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                },
            },
        },
    )

  def test_parse_instructions_schema_invalid(self):
    instructions = 'oops'
    schema = conftest.parse_instructions_schema(instructions)
    self.assertIsNone(schema)

  def test_parse_instructions_schema_malformed_json(self):
    instructions = textwrap.dedent("""
    ```
    {
      "invalid": json
    }
    ```
    """)
    schema = conftest.parse_instructions_schema(instructions)
    self.assertIsNone(schema)


class BaseFormattingInstructionsTest:
  """For testing formatting instructions produced by output schema models."""

  MODEL_CLASS = None  # to be set by subclasses
  EXPECTED_INTERVALS = None  # to be set by subclasses

  def setUp(self):
    super().setUp()
    self.model_class = self.MODEL_CLASS
    self.parser = PydanticOutputParser(pydantic_object=self.model_class)
    self.instructions = self.parser.get_format_instructions()
    self.schema = conftest.parse_instructions_schema(self.instructions)

  def test_formatting_instructions(self):
    self.assertIsInstance(self.instructions, str)
    self.assertIn(
        'The output should be formatted as a JSON instance that conforms to the'
        ' JSON schema below.',
        self.instructions,
    )

  def test_schema(self):
    self.assertIsInstance(self.schema, dict)
    self.assertCountEqual(
        self.schema.keys(),
        ['$defs', 'description', 'properties', 'required'],
    )


class FormattingInstructionsTest(BaseFormattingInstructionsTest, absltest.TestCase):  # pylint: disable=line-too-long

  MODEL_CLASS = output_schema.SetpointsAction
  EXPECTED_INTERVALS = list(output_schema.DEFAULT_VALIDITY_INTERVALS)

  def test_schema_required_fields(self):
    self.assertCountEqual(
        self.schema['required'],
        [
            'timestamp',
            'justification',
            'setpoints',
            'validity_interval',
        ],
    )

  def test_schema_properties(self):
    self.assertDictEqual(
        self.schema['properties'],
        {
            'setpoints': {
                'description': 'A list of setpoints.',
                'items': {'$ref': '#/$defs/DeviceSetpoint'},
                'title': 'Setpoints',
                'type': 'array',
            },
            'timestamp': {
                'description': output_schema.TIMESTAMP_DESCRIPTION,
                'title': 'Timestamp',
                'type': 'string',
            },
            'justification': {
                'description': output_schema.JUSTIFICATION_DESCRIPTION,
                'title': 'Justification',
                'type': 'string',
            },
            'validity_interval': {
                'description': output_schema.VALIDITY_INTERVAL_DESCRIPTION,
                'enum': self.EXPECTED_INTERVALS,
                'title': 'Validity Interval',
                'type': 'integer',
            },
        },
    )

  def test_schema_defs(self):
    self.assertListEqual(list(self.schema['$defs'].keys()), ['DeviceSetpoint'])

    schema_def = self.schema['$defs']['DeviceSetpoint']
    expected = {
        'description': (
            'A single device setpoint.\n\nA device is uniquely identified by'
            ' a composite key consisting of the device\nidentifier and the'
            ' setpoint name.\n\nAttributes:\n  device_id: The unique'
            " identifier of the device (e.g. 'boiler-123-xyz').\n "
            ' setpoint_name: The name of the setpoint (e.g.'
            " 'supply_water_temperature').\n  setpoint_value: The requested"
            ' value to be set (e.g. 120.0).\n  justification: The reason for'
            ' choosing this specific device setting.'
        ),
        'properties': {
            'device_id': {
                'description': 'The unique identifier of the device.',
                'title': 'Device Id',
                'type': 'string',
            },
            'setpoint_name': {
                'description': 'The name of the setpoint.',
                'title': 'Setpoint Name',
                'type': 'string',
            },
            'setpoint_value': {
                'description': 'The requested value to be set.',
                'title': 'Setpoint Value',
                'type': 'number',
            },
            'justification': {
                'description': (
                    'The reason for choosing this specific device setting.'
                ),
                'title': 'Justification',
                'type': 'string',
            },
        },
        'required': [
            'device_id',
            'setpoint_name',
            'setpoint_value',
            'justification',
        ],
        'title': 'DeviceSetpoint',
        'type': 'object',
    }
    self.assertDictEqual(schema_def, expected)


class CustomIntervalInstructionsTest(FormattingInstructionsTest):

  CUSTOM_INTERVALS = [5, 10, 15, 20]

  MODEL_CLASS = output_schema.create_action_model(
      custom_intervals=CUSTOM_INTERVALS
  )
  EXPECTED_INTERVALS = CUSTOM_INTERVALS


if __name__ == '__main__':
  absltest.main()
