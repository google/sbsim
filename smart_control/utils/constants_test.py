"""Tests for constants."""

import os

from absl.testing import absltest

from smart_control.utils.constants import ROOT_DIRPATH


class TestRelativePaths(absltest.TestCase):

  def test_root_dir(self):
    # test the path to the root directory is correct,
    # and some files that would only exist there are present

    file_names = os.listdir(ROOT_DIRPATH)
    self.assertIn("README.md", file_names)
    self.assertIn("pyproject.toml", file_names)
    self.assertIn("LICENSE", file_names)
    self.assertIn("METADATA", file_names)
    self.assertIn("OWNERS", file_names)


if __name__ == "__main__":
  absltest.main()
