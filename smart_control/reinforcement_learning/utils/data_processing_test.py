"""Tests for reinforcement learning data processing utils.

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

from absl.testing import absltest

from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import convert_celsius_to_kelvin
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import convert_kelvin_to_celsius


class TestTempConversions(absltest.TestCase):

  def test_c_to_k(self):
    self.assertEqual(convert_celsius_to_kelvin(0), 273.15)

  def test_k_to_c(self):
    self.assertEqual(convert_kelvin_to_celsius(273.15), 0)


if __name__ == '__main__':
  absltest.main()
