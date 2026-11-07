"""Tests for the Base LLM Service interface."""

from absl.testing import absltest
from smart_buildings.smart_control.llm.services import conftest


class LlmServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.service = conftest.create_fake_llm_service()

  def test_temperature(self):
    self.assertEqual(self.service.temperature, 0.0)

  def test_get_response(self):
    response = self.service.get_response(conftest.PROMPT_TEXT)
    self.assertEqual(response, conftest.RESPONSE_TEXT)


if __name__ == "__main__":
  absltest.main()
