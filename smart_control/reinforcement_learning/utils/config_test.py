"""Tests for reinforcement learning utils config."""

import os

from absl.testing import absltest

from smart_control.utils.constants import ROOT_DIR


class TestConfigPaths(absltest.TestCase):

  def test_root_dir(self):
    # test the path to the root directory is correct,
    # and some files that would only exist there are present

    file_names = os.listdir(ROOT_DIR)
    self.assertIn("README.md", file_names)
    self.assertIn("pyproject.toml", file_names)
    self.assertIn("LICENSE", file_names)
    self.assertIn("METADATA", file_names)
    self.assertIn("OWNERS", file_names)


if __name__ == "__main__":
  absltest.main()
