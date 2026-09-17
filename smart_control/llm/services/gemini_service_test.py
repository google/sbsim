"""Tests for Gemini LLM service."""

import unittest
from unittest import mock

from absl.testing import absltest
from google import genai
from smart_buildings.smart_control.llm.services import conftest
from smart_buildings.smart_control.llm.services.gemini_service import GeminiService  # pylint: disable=g-importing-member

FAKE_API_KEY = "not-a-real-api-key"


class GeminiServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.service = GeminiService(api_key=FAKE_API_KEY)

  def test_api_key(self):
    self.assertEqual(self.service.api_key, FAKE_API_KEY)

  def test_client(self):
    self.assertIsInstance(self.service.client, genai.Client)

  def test_temperature(self):
    self.assertEqual(self.service.temperature, 0.1)

  def test_generation_config(self):
    config = self.service.generation_config
    expected_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    self.assertEqual(config, expected_config)

  @unittest.skipUnless(conftest.TEST_GEMINI_SERVICE_LIVE, conftest.SKIP_REASON)
  def test_get_response(self):
    response = self.service.get_response(conftest.PROMPT_TEXT)
    self.assertIsInstance(response, str)

  def test_get_response_mocked(self):
    client = mock.create_autospec(genai.Client, instance=True)
    generate_content_response = mock.MagicMock()
    generate_content_response.text = conftest.RESPONSE_TEXT
    client.models.generate_content.return_value = generate_content_response

    service = GeminiService(api_key=FAKE_API_KEY, client=client)
    response = service.get_response(conftest.PROMPT_TEXT)
    self.assertIsInstance(response, str)
    self.assertEqual(response, conftest.RESPONSE_TEXT)


class MockedGeminiServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.service = conftest.create_mock_gemini_service()

  def test_get_response(self):
    response = self.service.get_response(conftest.PROMPT_TEXT)
    self.assertIsInstance(response, str)
    self.assertEqual(response, conftest.RESPONSE_TEXT)


if __name__ == "__main__":
  absltest.main()
