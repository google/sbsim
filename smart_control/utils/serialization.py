"""JSON-serialization utilities."""

from typing import Any
import pandas as pd

SerializableData = dict[str, Any]


def to_serializable(data: Any) -> Any:
  """Converts native types to JSON-serializable types.

  Handles dictionaries, lists, tuples, sets, pandas Timestamps, and Exceptions.

  Ensures data can be saved to a JSON file.

  Args:
    data: The data to convert.

  Returns:
    The data as JSON-serializable types.
  """
  if isinstance(data, dict):
    return {k: to_serializable(v) for k, v in data.items()}
  if isinstance(data, (list, tuple, set)):
    return [to_serializable(v) for v in data]
  if isinstance(data, pd.Timestamp):
    return str(data)
  if isinstance(data, Exception):
    return str(data)
  return data
