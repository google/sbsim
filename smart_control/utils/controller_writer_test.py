"""Tests for open source controller writer."""

import json
import os

from absl.testing import absltest
import pandas as pd
import plotly.graph_objects as go

from smart_buildings.smart_control.utils import controller_writer


class ControllerWriteGenericFilesTest(absltest.TestCase):
  """Tests for generic file-writing methods."""

  def setUp(self):
    super().setUp()
    self.working_dir = self.create_tempdir()
    self.writer = controller_writer.ProtoWriter(self.working_dir)

  def _open(self, filepath, mode):
    """Opens a file for reading. Can be overridden in corp codebase."""
    return self.writer._open(filepath, mode)

  def test_output_dir(self):
    """Tests that the output directory is set correctly."""
    self.assertEqual(self.writer.output_dir, self.working_dir)

  def test_write_txt(self):
    """Tests that the TXT file is written correctly."""
    filename = 'test.txt'
    txt = 'testing 1, 2, 3'
    self.writer.write_txt(txt, filename=filename)

    filepath = os.path.join(self.working_dir, filename)
    self.assertTrue(os.path.exists(filepath))
    with self._open(filepath, 'r') as f:
      self.assertEqual(f.read(), txt)

  def test_write_json(self):
    """Tests that the JSON file is written correctly."""
    json_data = {'testing': [1, 2, 3]}
    filename = 'test.json'
    self.writer.write_json(json_data, filename=filename)

    filepath = os.path.join(self.working_dir, filename)
    self.assertTrue(os.path.exists(filepath))
    with self._open(filepath, 'r') as f:
      self.assertEqual(f.read(), json.dumps(json_data, indent=2))

  def test_write_csv(self):
    """Tests that the CSV file is written correctly."""
    df = pd.DataFrame({'testing': [1, 2, 3]})
    filename = 'test.csv'
    self.writer.write_csv(df, filename=filename)

    filepath = os.path.join(self.working_dir, filename)
    self.assertTrue(os.path.exists(filepath))
    with self._open(filepath, 'r') as f:
      self.assertEqual(f.read(), df.to_csv(index=False))

  def test_write_plot_html(self):
    """Tests that the plot HTML file is written correctly."""
    fig = go.Figure()
    fig.update_layout(title='My Plot Title')

    filename = 'test.html'
    self.writer.write_plot_html(fig, filename=filename)

    filepath = os.path.join(self.working_dir, filename)
    self.assertTrue(os.path.exists(filepath))
    with self._open(filepath, 'r') as f:
      file_content = f.read()
      self.assertIn('<html>', file_content)
      self.assertIn('</html>', file_content)
      self.assertIn('Plotly.newPlot', file_content)
      self.assertIn('My Plot Title', file_content)
