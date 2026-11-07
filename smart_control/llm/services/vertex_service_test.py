"""Tests for Vertex AI LLM service."""

import unittest
from unittest import mock

from absl.testing import absltest
from google import genai
from google.auth import credentials
from smart_buildings.smart_control.llm.services import conftest
from smart_buildings.smart_control.llm.services import vertex_service


class VertexAIServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.service = vertex_service.VertexAIService()

  def test_project_id(self):
    self.assertEqual(self.service.project_id, 'smart-buildings-dev')

  def test_location(self):
    self.assertEqual(self.service.location, 'us-central1')

  def test_model_name(self):
    self.assertEqual(self.service.model_name, 'gemini-2.5-flash')

  def test_temperature(self):
    self.assertEqual(self.service.temperature, 0.1)

  @unittest.skipUnless(conftest.TEST_VERTEX_SERVICE_LIVE, conftest.SKIP_REASON)
  def test_credentials(self):
    self.assertIsInstance(self.service.credentials, credentials.Credentials)

  def test_client(self):
    self.assertIsInstance(self.service.client, genai.Client)

  def test_generation_config(self):
    config = self.service.generation_config
    self.assertIsInstance(config, dict)  # or genai.types.GenerationConfig
    expected_config = {
        'temperature': 0.1,
        'top_p': 0.95,
        'top_k': 40,
        'max_output_tokens': 1024,
        'safety_settings': vertex_service.SAFETY_DISABLED,
    }
    self.assertEqual(config, expected_config)

  @unittest.skipUnless(conftest.TEST_VERTEX_SERVICE_LIVE, conftest.SKIP_REASON)
  def test_get_response(self):
    response = self.service.get_response(conftest.PROMPT_TEXT)
    # non-deterministic result from real service, just checking the type:
    self.assertIsInstance(response, str)

  def test_get_response_mocked(self):
    # mocked credentials:
    creds = mock.create_autospec(credentials.Credentials, instance=True)

    # mocked client:
    client = mock.create_autospec(genai.Client, instance=True)
    generate_content_response = mock.MagicMock()
    generate_content_response.text = conftest.RESPONSE_TEXT
    client.models.generate_content.return_value = generate_content_response

    # dependency injection:
    service = vertex_service.VertexAIService(
        project_id='not-a-real-project', credentials=creds, client=client
    )

    # test the response:
    response = service.get_response(conftest.PROMPT_TEXT)
    self.assertIsInstance(response, str)
    self.assertEqual(response, conftest.RESPONSE_TEXT)


class MockedVertexAIServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.service = conftest.create_mock_vertex_service()

  def test_credentials(self):
    self.assertIsInstance(self.service.credentials, credentials.Credentials)

  def test_get_response(self):
    response = self.service.get_response(conftest.PROMPT_TEXT)
    self.assertEqual(response, conftest.RESPONSE_TEXT)


if __name__ == '__main__':
  absltest.main()
