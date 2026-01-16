"""Comparison script for occupancy models.

This script generates comparison charts showing the behavior of all occupancy
models over a 24-hour period. The charts are saved as both PNG (for documentation)
and HTML (for interactive viewing).
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from smart_control.simulator.occupancy import EnhancedOccupancy
from smart_control.simulator.occupancy import LIGHTSWITCHOccupancy
from smart_control.simulator.occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.occupancy import StepFunctionOccupancy


def create_occupancy_models() -> Dict[str, object]:
  """Creates instances of all occupancy models with reasonable defaults.

  Returns:
    Dictionary mapping model names to model instances.
  """
  models = {}

  # Step Function Occupancy - simple constant levels
  models['Step Function'] = StepFunctionOccupancy(
      work_start_time=pd.Timedelta(9, unit='h'),
      work_end_time=pd.Timedelta(17, unit='h'),
      work_occupancy=10.0,
      nonwork_occupancy=0.5,
  )

  # Randomized Arrival/Departure Occupancy
  models['Randomized Arrival/Departure'] = (
      RandomizedArrivalDepartureOccupancy(
          zone_assignment=10,
          earliest_expected_arrival_hour=8,
          latest_expected_arrival_hour=10,
          earliest_expected_departure_hour=16,
          latest_expected_departure_hour=18,
          time_step_sec=300,  # 5 minutes
          seed=42,
          time_zone='US/Pacific',
      )
  )

  # LIGHTSWITCH Occupancy (Stochastic)
  models['LIGHTSWITCH (Stochastic)'] = LIGHTSWITCHOccupancy(
      zone_assignment=10,
      earliest_expected_arrival_hour=8,
      latest_expected_arrival_hour=10,
      earliest_expected_departure_hour=16,
      latest_expected_departure_hour=18,
      lunch_start_hour=12,
      lunch_end_hour=13,
      time_step_sec=300,  # 5 minutes
      seed=42,
      time_zone='US/Pacific',
  )

  # Enhanced Occupancy - with minute-level control
  models['Enhanced (Minute-Level)'] = EnhancedOccupancy(
      zone_assignment=10,
      earliest_expected_arrival_hour=8,
      latest_expected_arrival_hour=10,
      earliest_expected_departure_hour=16,
      latest_expected_departure_hour=18,
      lunch_start_hour=12,
      lunch_end_hour=13,
      time_step=pd.Timedelta(5, unit='min'),
      time_zone='US/Pacific',
  )

  return models


def generate_occupancy_data(
    models: Dict[str, object], zone_id: str = 'zone_0'
) -> pd.DataFrame:
  """Generates occupancy data for all models over a 24-hour period.

  Args:
    models: Dictionary of occupancy model instances.
    zone_id: Zone identifier to query.

  Returns:
    DataFrame with timestamp index and columns for each model.
  """
  # Create timestamps for a single day in 5-minute intervals
  # Use a weekday (Monday) to ensure work patterns are active
  start_time = pd.Timestamp('2024-01-08 00:00:00', tz='US/Pacific')
  end_time = start_time + pd.Timedelta(24, unit='h')
  time_step = pd.Timedelta(5, unit='min')

  timestamps = pd.date_range(start=start_time, end=end_time, freq=time_step)

  # Calculate occupancy for each model at each timestamp
  data = {'Timestamp': timestamps[:-1]}  # Exclude the last timestamp

  for model_name, model in models.items():
    occupancies = []
    for i in range(len(timestamps) - 1):
      interval_start = timestamps[i]
      interval_end = timestamps[i + 1]
      try:
        occupancy = model.average_zone_occupancy(
            zone_id, interval_start, interval_end
        )
        occupancies.append(occupancy)
      except Exception as e:
        print(
            f'Error calculating occupancy for {model_name} at'
            f' {interval_start}: {e}'
        )
        occupancies.append(0.0)

    data[model_name] = occupancies

  df = pd.DataFrame(data)
  df.set_index('Timestamp', inplace=True)
  return df


def create_matplotlib_plot(df: pd.DataFrame, output_path: str):
  """Creates a matplotlib plot and saves as PNG.

  Args:
    df: DataFrame with occupancy data.
    output_path: Path to save the PNG file.
  """
  plt.figure(figsize=(14, 7))

  for column in df.columns:
    # Convert timestamps to local time strings for x-axis
    times = df.index.strftime('%H:%M')
    plt.plot(range(len(df)), df[column], label=column, linewidth=2)

  plt.xlabel('Time of Day (Pacific Time)', fontsize=12)
  plt.ylabel('Average Occupancy (people)', fontsize=12)
  plt.title(
      'Occupancy Model Comparison - 24 Hour Period (5-Minute Intervals)',
      fontsize=14,
      fontweight='bold',
  )
  plt.legend(loc='best', fontsize=10)
  plt.grid(True, alpha=0.3)

  # Set x-axis ticks to show every 2 hours
  tick_indices = range(0, len(df), 24)  # Every 2 hours (24 * 5 min = 2 hours)
  tick_labels = [df.index[i].strftime('%H:%M') for i in tick_indices]
  plt.xticks(tick_indices, tick_labels, rotation=45)

  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  print(f'Saved PNG plot to: {output_path}')


def create_plotly_plot(df: pd.DataFrame, output_path: str):
  """Creates an interactive plotly plot and saves as HTML.

  Args:
    df: DataFrame with occupancy data.
    output_path: Path to save the HTML file.
  """
  fig = go.Figure()

  for column in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column,
            line=dict(width=2),
        )
    )

  fig.update_layout(
      title={
          'text': (
              'Occupancy Model Comparison - 24 Hour Period (5-Minute'
              ' Intervals)'
          ),
          'x': 0.5,
          'xanchor': 'center',
          'font': {'size': 16, 'weight': 'bold'},
      },
      xaxis_title='Time of Day (Pacific Time)',
      yaxis_title='Average Occupancy (people)',
      hovermode='x unified',
      template='plotly_white',
      width=1200,
      height=600,
      legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
  )

  fig.update_xaxes(
      tickformat='%H:%M',
      dtick=7200000,  # 2 hours in milliseconds
  )

  fig.write_html(output_path)
  print(f'Saved HTML plot to: {output_path}')


def main():
  """Main function to generate comparison charts."""
  # Get the project root directory
  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.join(script_dir, '..', '..', '..')

  # Define output paths
  png_output = os.path.join(
      project_root, 'docs', 'assets', 'images', 'occupancy_comparison.png'
  )
  html_output = os.path.join(
      project_root, 'docs', 'assets', 'plots', 'occupancy_comparison.html'
  )

  # Ensure output directories exist
  os.makedirs(os.path.dirname(png_output), exist_ok=True)
  os.makedirs(os.path.dirname(html_output), exist_ok=True)

  print('Creating occupancy models...')
  models = create_occupancy_models()

  print('Generating occupancy data...')
  df = generate_occupancy_data(models)

  print('Creating matplotlib plot...')
  create_matplotlib_plot(df, png_output)

  print('Creating plotly plot...')
  create_plotly_plot(df, html_output)

  print('\\nComparison charts generated successfully!')
  print(f'PNG: {png_output}')
  print(f'HTML: {html_output}')


if __name__ == '__main__':
  main()
