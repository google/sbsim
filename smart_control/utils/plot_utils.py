"""Utilities for plotting simulation and converting to video.

Copyright 2022 Google LLC

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

import collections
import os
import pathlib

from matplotlib import cm
from matplotlib import patches
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

K_TO_C = 273.0  # TODO: https://github.com/google/sbsim/issues/25 - consider importing and using `int(KELVIN_TO_CELSIUS)` constant here # pylint:disable=line-too-long


def get_temp_colors(min_k, max_k):
  """Returns a color gradient for the temps between min and max_k.

  Temperatures are measured in Kelvin.

  Args:
    min_k: min temp in kelvin
    max_k: max temp in kelvin
  """

  def get_temp_color_pallet(num_colors):
    x = np.arange(num_colors)
    ys = [i + x + (i * x) ** 2 for i in range(num_colors)]
    return cm.get_cmap('rainbow')(np.linspace(0, 1, len(ys)))

  num_colors = max_k - min_k + 1
  colors = get_temp_color_pallet(num_colors)
  temp_color_map = {}
  temp = min_k
  for c in colors:
    temp_color_map[temp] = c
    temp += 1
  return temp_color_map


def render_building_subplot(
    ax, min_k, max_k, ambient_temp, building, current_time
):
  """Renders the planform view of the building."""

  building_width_x = ((building.cv_size_cm + 2) / 100.0) * (
      building.room_shape[0] + 1
  ) * building.building_shape[0] + 2
  building_width_y = ((building.cv_size_cm + 2) / 100.0) * (
      building.room_shape[1] + 1
  ) * building.building_shape[1] + 2
  zone_width_x = building.room_shape[0] + 1
  zone_width_y = building.room_shape[1] + 1
  denom_x = building_width_x
  denom_y = building_width_y
  delta_x = building.cv_size_cm / 100.0

  temp_color_map = get_temp_colors(min_k, max_k)

  def get_temp_color(temp):
    """Bounds the colors within min_k and max_k."""
    if temp < min_k:
      temp_color = temp_color_map[min_k]
    elif temp > max_k:
      temp_color = temp_color_map[max_k]
    else:
      temp_color = temp_color_map[int(temp)]
    return temp_color

  def render_ambient(temp):
    """Draws an exterior rectangle around the building based on ambient temp."""

    temp_color = get_temp_color(temp)
    width = (
        ((building.room_shape[0] + 1) * building.building_shape[0] + 3)
        * delta_x
        / denom_x
    )
    height = (
        ((building.room_shape[1] + 1) * building.building_shape[1] + 3)
        * delta_x
        / denom_y
    )

    p = plt.Rectangle(
        (0, 0),
        width,
        height,
        fill=True,
        edgecolor=None,
        alpha=0.6,
        facecolor=temp_color,
    )
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

    left = delta_x / denom_x / 2.0
    bottom = delta_x / denom_y / 2.0
    width = (
        ((building.room_shape[0] + 1) * building.building_shape[0] + 2)
        * delta_x
        / denom_x
    )
    height = (
        ((building.room_shape[1] + 1) * building.building_shape[1] + 2)
        * delta_x
        / denom_y
    )

    p = plt.Rectangle(
        (left, bottom),
        width,
        height,
        fill=True,
        edgecolor=None,
        alpha=1.0,
        facecolor='white',
    )
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

  def render_control_volume(i, j, temp, conductivity):
    """Renders the control volume facet at i, j based on the temp."""

    temp_color = get_temp_color(temp)
    if conductivity < 0.1:
      edgecolor = 'black'
    elif conductivity < 5.0:
      edgecolor = 'dimgray'
    else:
      edgecolor = 'lightgray'

    if i == 0:
      # left edge
      left = delta_x / denom_x / 2.0
      width = delta_x / denom_x / 2.0

    elif i == (building.room_shape[0] + 1) * building.building_shape[0] + 2:
      # right edge
      left = i * delta_x / denom_x
      width = delta_x / denom_x / 2.0

    else:
      # width interior
      left = i * delta_x / denom_x
      width = delta_x / denom_x

    if j == 0:
      # bottom edge
      bottom = delta_x / denom_y / 2.0
      height = delta_x / denom_y / 2.0

    elif j == (building.room_shape[1] + 1) * building.building_shape[1] + 2:
      # top edge
      bottom = j * delta_x / denom_y
      height = delta_x / denom_y / 2.0

    else:
      # height interior
      bottom = j * delta_x / denom_y
      height = delta_x / denom_y

    p = plt.Rectangle(
        (left, bottom),
        width,
        height,
        fill=True,
        edgecolor=edgecolor,
        alpha=0.6,
        facecolor=temp_color,
    )
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

  def render_zone(zi, zj):
    """Writes  temperature info about each zone."""

    left = (zi * zone_width_x * delta_x + delta_x) / denom_x
    bottom = (zj * zone_width_y * delta_x + delta_x) / denom_y
    height = zone_width_y * delta_x / denom_y

    temp_min, temp_max, temp_avg = building.get_zone_temp_stats((zi, zj))

    temp_label = (
        f'({zi}, {zj}) '
        f'min {(temp_min - K_TO_C):3.1f} C, '
        f'max {(temp_max - K_TO_C):3.1f} C, '
        f'avg {(temp_avg - K_TO_C):3.1f} C'
    )

    ax.text(
        0.01 + left,
        bottom + height - 0.017,
        temp_label,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=12,
    )

  def render_diffuser(i, j, q):
    """Draw the diffuser state."""
    height = 0.5 * delta_x / denom_y
    width = 0.5 * delta_x / denom_x

    x = (i + 0.5) * delta_x / denom_x
    y = (j + 0.5) * delta_x / denom_y

    if building.input_q[i][j] == 0.0:
      facecolor = 'gray'
    elif building.input_q[i][j] < 0.0:
      facecolor = 'blue'
    else:
      facecolor = 'red'
    p = patches.Ellipse(
        (x, y),
        height=height,
        width=width,
        facecolor=facecolor,
        alpha=1.0,
        edgecolor='gray',
    )
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

    if q > 0.0 or q < 0.0:
      ax.text(
          x + 0.005,
          y,
          f'{(q / 1000.0):3.1f} kW',
          horizontalalignment='left',
          verticalalignment='top',
          transform=ax.transAxes,
          fontsize=12,
      )

  render_ambient(ambient_temp)

  max_conductivity = building.conductivity.max()
  for i in range((building.room_shape[0] + 1) * building.building_shape[0] + 3):
    for j in range(
        (building.room_shape[1] + 1) * building.building_shape[1] + 3
    ):
      if building.conductivity[i][j] == max_conductivity:
        temp = building.temp[i][j]
        render_control_volume(i, j, temp, building.conductivity[i][j])

  for i in range((building.room_shape[0] + 1) * building.building_shape[0] + 3):
    for j in range(
        (building.room_shape[1] + 1) * building.building_shape[1] + 3
    ):
      if building.conductivity[i][j] < max_conductivity:
        temp = building.temp[i][j]
        render_control_volume(i, j, temp, building.conductivity[i][j])

  for zi in range(building.building_shape[0]):
    for zj in range(building.building_shape[1]):
      render_zone(zi, zj)

  # render_sources
  for i in range((building.room_shape[0] + 1) * building.building_shape[0] + 3):
    for j in range(
        (building.room_shape[1] + 1) * building.building_shape[1] + 3
    ):
      if building.diffusers[i][j] > 0:
        render_diffuser(i, j, building.diffusers[i][j] * building.input_q[i][j])

  label = (
      f"Local time {current_time.strftime('%Y-%m-%d %H:%M')}, "
      f'Ambient temp {(ambient_temp - K_TO_C):3.1f} C'
  )
  ax.text(
      0.01,
      1.0,
      label,
      horizontalalignment='left',
      verticalalignment='top',
      transform=ax.transAxes,
      fontsize=16,
  )
  ax.axis('off')


def plot_zone_temp_timeline(ax1, schedule, temps_timeseries_df, end_timestamp):
  """Plots timeline of zone temperature."""
  setpoint_windows = schedule.get_plot_data(
      temps_timeseries_df.index.min(), end_timestamp
  )
  for _, row in setpoint_windows.iterrows():
    left = mdates.date2num(row['start_time'])
    bottom = row['heating_setpoint'] - K_TO_C
    width = mdates.date2num(row['end_time']) - left
    height = row['cooling_setpoint'] - row['heating_setpoint']
    face_color = 'white'
    p = plt.Rectangle(
        (left, bottom),
        width,
        height,
        fill=True,
        edgecolor=None,
        alpha=0.3,
        facecolor=face_color,
    )

    ax1.add_patch(p)

  zone_temps_cols = list(set(temps_timeseries_df.columns) - {'ambient'})
  for zone in zone_temps_cols:
    ax1.plot(
        temps_timeseries_df.index,
        temps_timeseries_df[zone] - K_TO_C,
        color='yellow',
        marker=None,
        alpha=1,
        lw=1,
        linestyle='-',
    )

  ax1.plot(
      temps_timeseries_df.index,
      temps_timeseries_df['ambient'] - K_TO_C,
      color='blue',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='-',
  )
  ax1.set_facecolor('black')
  ax1.xaxis.tick_top()
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d %H:%M'))
  ax1.grid(color='gray', linestyle='-', linewidth=0.7)
  ax1.set_ylabel('Temp [C]', color='blue', fontsize=14)
  ax1.set_xlim(left=temps_timeseries_df.index.min(), right=end_timestamp)
  ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_energy_rates_timeline(ax1, energy_rates_df, end_timestamp):
  """Plots timeline of energy rates."""

  ax1.plot(
      energy_rates_df.index,
      energy_rates_df['boiler_thermal_energy_rate'] / 1000.0,
      color='lime',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='-',
  )
  ax1.plot(
      energy_rates_df.index,
      energy_rates_df['boiler_electrical_energy_rate'] / 1000.0,
      color='lime',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='--',
  )

  ax1.plot(
      energy_rates_df.index,
      energy_rates_df['air_handler_intake_fan_energy_rate'] / 1000.0,
      color='magenta',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='--',
  )
  ax1.plot(
      energy_rates_df.index,
      energy_rates_df['air_handler_exhaust_fan_energy_rate'] / 1000.0,
      color='magenta',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='--',
  )
  ax1.plot(
      energy_rates_df.index,
      energy_rates_df['air_handler_thermal_energy_rate'] / 1000.0,
      color='magenta',
      marker=None,
      alpha=1,
      lw=3,
      linestyle='-',
  )
  ax1.set_facecolor('black')
  ax1.xaxis.tick_bottom()
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d %H:%M'))
  ax1.grid(color='gray', linestyle='-', linewidth=0.7)
  ax1.set_ylabel('Energy Rate [kW]', color='blue', fontsize=14)
  ax1.set_xlim(left=energy_rates_df.index.min(), right=end_timestamp)
  ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_combined_results(
    temps_timeseries_df,
    energy_rates_df,
    min_k,
    max_k,
    ambient_temp,
    building,
    schedule,
    current_time,
    end_timestamp,
    writedir=None,
):
  """Plot results of building, zone temp, and energy."""
  fig, (ax1, ax2, ax3) = plt.subplots(
      nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 1, 2.3]}, squeeze=True
  )
  fig.set_size_inches(40, 40)

  plot_zone_temp_timeline(ax1, schedule, temps_timeseries_df, end_timestamp)

  plot_energy_rates_timeline(ax2, energy_rates_df, end_timestamp)

  render_building_subplot(
      ax3, min_k, max_k, ambient_temp, building, current_time
  )

  if writedir:

    filename = f"thermal_step_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    full_path = os.path.join(writedir, filename)
    full_path = pathlib.Path(full_path)

    with full_path.open(mode='wb') as fd:
      plt.savefig(fd)
  plt.show()


def init_metrics():
  """Initializes the metrics for sumlation."""

  metrics = {}
  metrics['timestamps'] = []
  metrics['ambient_temps'] = []
  metrics['avg_temps_timeseries'] = collections.defaultdict(list)
  metrics['boiler_thermal_energy_rates'] = []
  metrics['boiler_electrical_energy_rates'] = []
  metrics['air_handler_intake_fan_energy_rates'] = []
  metrics['air_handler_exhaust_fan_energy_rates'] = []
  metrics['air_handler_thermal_energy_rates'] = []
  return metrics


def update_metrics(
    metrics,
    current_timestamp,
    current_ambient_temp,
    supply_air_temp,
    hvac,
    recirculation_temp,
):
  """Updates the metrics on sim update."""
  metrics['timestamps'].append(current_timestamp)
  metrics['ambient_temps'].append(current_ambient_temp)
  metrics['boiler_thermal_energy_rates'].append(
      hvac.hot_water_system.compute_thermal_energy_rate(
          return_water_temp=supply_air_temp,
          outside_temp=hvac.hot_water_system.reheat_water_setpoint,
      )
  )
  metrics['boiler_electrical_energy_rates'].append(
      hvac.hot_water_system.compute_pump_power() * 1000
  )  # TODO(judahg) verify this is correct
  metrics['air_handler_intake_fan_energy_rates'].append(
      hvac.air_handler.compute_intake_fan_energy_rate()
  )
  metrics['air_handler_exhaust_fan_energy_rates'].append(
      hvac.air_handler.compute_exhaust_fan_energy_rate()
  )
  metrics['air_handler_thermal_energy_rates'].append(
      hvac.air_handler.compute_thermal_energy_rate(
          current_ambient_temp, recirculation_temp
      )
  )
  return metrics


def plot_update(
    metrics,
    current_ambient_temp,
    building,
    schedule,
    current_timestamp,
    end_timestamp,
    img_dir,
):
  """Plots the temp timeline, energy rate timeline, and thermal view."""

  temps_timeseries_df = pd.DataFrame(index=metrics['timestamps'])
  temps_timeseries_df['ambient'] = metrics['ambient_temps']

  for zone in metrics['avg_temps_timeseries']:
    temps_timeseries_df[zone] = metrics['avg_temps_timeseries'][zone]

  energy_rates_df = pd.DataFrame(
      {
          'boiler_thermal_energy_rate': metrics['boiler_thermal_energy_rates'],
          'boiler_electrical_energy_rate': metrics[
              'boiler_electrical_energy_rates'
          ],
          'air_handler_intake_fan_energy_rate': metrics[
              'air_handler_intake_fan_energy_rates'
          ],
          'air_handler_exhaust_fan_energy_rate': metrics[
              'air_handler_exhaust_fan_energy_rates'
          ],
          'air_handler_thermal_energy_rate': metrics[
              'air_handler_thermal_energy_rates'
          ],
      },
      index=metrics['timestamps'],
  )

  plot_combined_results(
      temps_timeseries_df,
      energy_rates_df,
      280,
      300,
      current_ambient_temp,
      building,
      schedule,
      current_timestamp,
      end_timestamp,
      img_dir,
  )
