"""Tests for reinforcement learning utils config.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

from absl.testing import absltest

from smart_buildings.smart_control.reinforcement_learning.utils.config import ROOT_DIR


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
