import json

from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.utils import serialization


class SerializationTest(absltest.TestCase):

  def test_to_serializable_dict(self):
    data = {"a": 1, "b": [2, 3]}
    result = serialization.to_serializable(data)
    self.assertEqual(result, data)
    json.dumps(result)

  def test_to_serializable_list(self):
    data = [1, 2, 3]
    result = serialization.to_serializable(data)
    self.assertEqual(result, data)
    json.dumps(result)

  def test_to_serializable_tuple(self):
    data = (1, 2, 3)
    result = serialization.to_serializable(data)
    self.assertEqual(result, [1, 2, 3])
    json.dumps(result)

  def test_to_serializable_set(self):
    data = {1, 2, 3}
    result = serialization.to_serializable(data)
    self.assertCountEqual(result, [1, 2, 3])
    json.dumps(result)

  def test_to_serializable_timestamp(self):
    data = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    result = serialization.to_serializable(data)
    self.assertEqual(
        result, "2026-01-01 00:00:00+00:00"
    )
    json.dumps(result)

  def test_to_serializable_exception(self):
    data = ValueError("test error")
    result = serialization.to_serializable(data)
    self.assertEqual(result, "test error")
    json.dumps(result)

  def test_to_serializable_nested(self):
    data = {
        "error": ValueError("boom"),
        "times": [pd.Timestamp("2026-01-01", tz="UTC")],
        "others": {1, 2},
    }
    result = serialization.to_serializable(data)
    self.assertEqual(result["error"], "boom")
    self.assertEqual(result["times"], ["2026-01-01 00:00:00+00:00"])
    self.assertCountEqual(result["others"], [1, 2])
    json.dumps(result)

  def test_to_serializable_unmodified(self):
    result = serialization.to_serializable(1.5)
    self.assertEqual(result, 1.5)
    json.dumps(result)


if __name__ == "__main__":
  absltest.main()
