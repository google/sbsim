import json

from absl.testing import absltest
from smart_buildings.smart_control.llm.agents import base_agent


class ErrorRecordNestedExceptionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    nested_error = ValueError('Something went wrong')

    self.record = base_agent.AgentErrorRecord(
        error_type='ValidationError',
        error_message='Validation failed',
        details=[{
            'loc': ('field',),
            'ctx': {'error': nested_error},
        }],
        metadata={'current_step': 4, 'response_txt': 'OOPS'},
    )

  def test_json_metadata(self):
    self.assertEqual(
        self.record.json_metadata,
        {
            'error_type': 'ValidationError',
            'error_message': 'Validation failed',
            'details': [{
                'loc': ['field'],
                'ctx': {'error': 'Something went wrong'},
            }],
            'metadata': {'current_step': 4, 'response_txt': 'OOPS'},
        },
    )

  def test_json_metadata_is_serializable(self):
    self.assertIsInstance(json.dumps(self.record.json_metadata), str)


if __name__ == '__main__':
  absltest.main()
