import os

from absl.testing import absltest

from smart_buildings.smart_control.utils import constants


class ConstantsTest(absltest.TestCase):

  def test_repo_dirpath(self):
    self.assertTrue(os.path.exists(constants.REPO_DIRPATH))

    with self.subTest(name="contents"):
      contents = os.listdir(constants.REPO_DIRPATH)
      self.assertIn("README.md", contents)
      self.assertIn("BUILD", contents)
      self.assertIn("agents", contents)
      self.assertIn("configs", contents)
      self.assertIn("dataset", contents)
      self.assertIn("environment", contents)
      self.assertIn("llm", contents)
      self.assertIn("models", contents)
      self.assertIn("notebooks", contents)
      self.assertIn("proto", contents)
      self.assertIn("reinforcement_learning", contents)
      self.assertIn("reward", contents)
      self.assertIn("services", contents)
      self.assertIn("simulator", contents)
      self.assertIn("utils", contents)


