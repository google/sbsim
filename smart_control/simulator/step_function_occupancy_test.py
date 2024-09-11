"""Tests for step_function_occupancy_model.

Copyright 2023 Google LLC

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
from absl.testing import parameterized
import pandas as pd
from smart_control.simulator import step_function_occupancy


class StepFunctionOccupancyModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "full day",
          pd.Timedelta("00:00:00"),
          pd.Timedelta(25, unit="hour"),
          (8 * 3600.0, 8 * 3600.0, 8 * 3600.0),
      ),
      (
          "workday, 20 min",
          pd.Timedelta("15:10:00"),
          pd.Timedelta("15:30:00"),
          (0.0, 1200.0, 0.0),
      ),
      (
          "before workday, 30 min",
          pd.Timedelta("03:15:00"),
          pd.Timedelta("03:45:00"),
          (1800.0, 0.0, 0.0),
      ),
      (
          "after workday, 120 min",
          pd.Timedelta("18:15:00"),
          pd.Timedelta("20:15:00"),
          (0.0, 0.0, 7200.0),
      ),
      (
          "during 20 min, after workday 120 min",
          pd.Timedelta("15:40:00"),
          pd.Timedelta("18:00:00"),
          (0.0, 1200.0, 7200.0),
      ),
      (
          "before 1 min, during workday 1 min",
          pd.Timedelta("07:59:00"),
          pd.Timedelta("08:01:00"),
          (60.0, 60.0, 0.0),
      ),
  )
  def test_split_workday_valid(self, start_time, end_time, expected_split):
    occupancy = step_function_occupancy.StepFunctionOccupancy(
        pd.Timedelta("08:00:00"), pd.Timedelta("16:00:00"), 10.0, 0.2
    )
    split = occupancy._split_workday(start_time, end_time)
    self.assertEqual(expected_split, split)

  def test_split_workday_invalid(self):
    occupancy = step_function_occupancy.StepFunctionOccupancy(
        pd.Timedelta("08:00:00"), pd.Timedelta("16:00:00"), 10.0, 0.2
    )
    with self.assertRaises(ValueError):
      _ = occupancy._split_workday(
          pd.Timedelta("10:00:00"), pd.Timedelta("09:59:00")
      )

    with self.assertRaises(ValueError):
      _ = occupancy._split_workday(
          pd.Timedelta(25, unit="hours"), pd.Timedelta("09:59:00")
      )

  @parameterized.named_parameters(
      (
          "full day",
          pd.Timestamp("2021-05-10 00:00:00"),
          pd.Timestamp("2021-05-11 00:00:00"),
          3.46666,
      ),
      (
          "two days",
          pd.Timestamp("2021-05-10 00:00:00"),
          pd.Timestamp("2021-05-12 00:00:00"),
          3.46666,
      ),
      (
          "full day holiday",
          pd.Timestamp("2022-12-25 00:00:00"),
          pd.Timestamp("2022-12-26 00:00:00"),
          0.2,
      ),
      (
          "full day weekend",
          pd.Timestamp("2021-05-09 00:00:00"),
          pd.Timestamp("2021-05-10 00:00:00"),
          0.2,
      ),
      (
          "split before",
          pd.Timestamp("2021-05-10 06:00:00"),
          pd.Timestamp("2021-05-10 10:00:00"),
          5.1,
      ),
      (
          "split after",
          pd.Timestamp("2021-05-10 15:55:00"),
          pd.Timestamp("2021-05-10 16:05:00"),
          5.1,
      ),
      (
          "before only",
          pd.Timestamp("2021-05-10 07:55:00"),
          pd.Timestamp("2021-05-10 07:56:00"),
          0.2,
      ),
      (
          "after only",
          pd.Timestamp("2021-05-10 17:55:00"),
          pd.Timestamp("2021-05-10 17:56:00"),
          0.2,
      ),
      (
          "during only",
          pd.Timestamp("2021-05-10 13:16:00"),
          pd.Timestamp("2021-05-10 13:22:00"),
          10.0,
      ),
      (
          "between years",
          pd.Timestamp("2022-12-30 18:00:00"),
          pd.Timestamp("2023-01-01 10:00"),
          0.2,
      ),
  )
  def test_average_zone_occupancy(
      self, start_time, end_time, expected_occupancy
  ):
    occupancy = step_function_occupancy.StepFunctionOccupancy(
        pd.Timedelta("08:00:00"), pd.Timedelta("16:00:00"), 10.0, 0.2
    )
    average_occupancy = occupancy.average_zone_occupancy(
        "any zone", start_time, end_time
    )
    self.assertAlmostEqual(expected_occupancy, average_occupancy, places=4)

  def test_invalid_average_zone_occupancy(self):
    occupancy = step_function_occupancy.StepFunctionOccupancy(
        pd.Timedelta("08:00:00"), pd.Timedelta("16:00:00"), 10.0, 0.2
    )
    with self.assertRaises(ValueError):
      _ = occupancy.average_zone_occupancy(
          "any zone",
          pd.Timestamp("2021-12-30 14:00:00"),
          pd.Timestamp("2021-12-30 13:00:00"),
      )

  def test_invalid_average_zone_occupancy_init(self):
    with self.assertRaises(ValueError):
      _ = step_function_occupancy.StepFunctionOccupancy(
          pd.Timedelta("-08:00:00"), pd.Timedelta("16:00:00"), 10.0, 0.2
      )

    with self.assertRaises(ValueError):
      _ = step_function_occupancy.StepFunctionOccupancy(
          pd.Timedelta("08:00:00"), pd.Timedelta("-16:00:00"), 10.0, 0.2
      )


if __name__ == "__main__":
  absltest.main()
