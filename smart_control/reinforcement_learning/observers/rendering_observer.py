"""Observer for rendering and visualizing environments.

This module provides an observer for rendering RL environments and visualizing
agent behavior through plots.
"""

import logging
import os
from typing import Callable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import pytz
from tf_agents.trajectories import trajectory as trajectory_lib

from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_buildings.smart_control.reinforcement_learning.utils.config import RENDERS_PATH
from smart_buildings.smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_buildings.smart_control.reinforcement_learning.utils.constants import KELVIN_TO_CELSIUS as _KELVIN_TO_CELSIUS
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_action_timeseries
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_energy_timeseries
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_latest_episode_reader
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_outside_air_temperature_timeseries
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_reward_timeseries
from smart_buildings.smart_control.reinforcement_learning.utils.data_processing import get_zone_timeseries
from smart_buildings.smart_control.utils import building_renderer

logger = logging.getLogger(__name__)


class RenderingObserver(Observer):
  """Observer that renders the environment and plots metrics.

  This observer renders the environment at specified intervals and can
  also show plots of metrics.
  """

  # Class constant
  KELVIN_TO_CELSIUS = _KELVIN_TO_CELSIUS

  def __init__(
      self,
      render_interval_steps: int = 10,
      env=None,  # consider: `Optional[environment.Environment] = None`
      render_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      plot_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      clear_output_before_render: bool = True,
      time_zone: str = DEFAULT_TIME_ZONE,
      save_path: str = RENDERS_PATH,
  ):
    """Initialize the observer.

    Args:
        render_interval_steps: Number of steps between renders.
        env: The environment to render. This must support the
          current_simulation_timestamp property if plot_fn is specified.
        render_fn: Optional function to use for rendering. If not provided,
          environment.render() will be used.
        plot_fn: Optional function to use for plotting. If not provided, no
          plotting will be done.
        clear_output_before_render: Whether to clear output before rendering.
        time_zone: Time zone for plotting timestamps.
        save_path: Directory path to save rendered visualizations.
    """
    self._counter = 0
    self._render_interval_steps = render_interval_steps
    self._environment = env
    self._render_fn = render_fn
    self._plot_fn = plot_fn
    self._clear_output_before_render = clear_output_before_render
    self._time_zone = time_zone
    self._cumulative_reward = 0.0
    self._start_time = None
    self._save_path = save_path

    # Create save directory if it doesn't exist
    os.makedirs(self._save_path, exist_ok=True)

    if self._environment is not None:
      # Store environment properties if available
      env = self._environment.pyenv.envs[0]
      if hasattr(env, '_num_timesteps_in_episode'):
        self._num_timesteps_in_episode = env._num_timesteps_in_episode

  def _format_plot(
      self, ax1, xlabel: str, start_time: int, end_time: int, time_zone: str
  ):
    """Formats a plot with common attributes."""
    ax1.set_facecolor('black')
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter('%a %m/%d %H:%M', tz=pytz.timezone(time_zone))
    )
    ax1.grid(color='gray', linestyle='-', linewidth=1.0)
    ax1.set_ylabel(xlabel, color='blue', fontsize=12)
    ax1.set_xlim(left=start_time, right=end_time)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(prop={'size': 10})

  def _plot_reward_timeline(self, ax1, reward_timeseries, time_zone):
    """Plot reward timeline."""

    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    ax1.plot(
        local_times,
        reward_timeseries['cumulative_reward'],
        color='royalblue',
        marker=None,
        alpha=1,
        lw=6,
        linestyle='-',
        label='reward',
    )
    self._format_plot(
        ax1,
        'Agent Reward',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )

  def _plot_energy_timeline(
      self, ax1, energy_timeseries, time_zone, cumulative=False
  ):
    """Plot energy timeline."""

    def _to_kwh(
        energy_rate: float,
        step_interval: pd.Timedelta = pd.Timedelta(5, unit='minute'),
    ) -> float:
      """Convert to kwh."""
      kw_power = energy_rate / 1000.0
      hwh_power = kw_power * step_interval / pd.Timedelta(1, unit='hour')
      return hwh_power.cumsum()

    # fmt: off
    # pylint: disable=line-too-long
    timeseries = energy_timeseries[energy_timeseries['device_type'] == 'air_handler']
    if cumulative:
      feature_timeseries_ac = _to_kwh(timeseries['air_handler_air_conditioner_energy_rate'])
      feature_timeseries_blower = _to_kwh(timeseries['air_handler_blower_electrical_energy_rate'])
    else:
      feature_timeseries_ac = timeseries['air_handler_air_conditioner_energy_rate'] / 1000.0
      feature_timeseries_blower = timeseries['air_handler_blower_electrical_energy_rate'] / 1000.0
    # pylint: enable=line-too-long
    # fmt: on

    ax1.plot(
        timeseries['start_time'],
        feature_timeseries_ac,
        color='magenta',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='-',
        label='AHU Electricity',
    )

    ax1.plot(
        timeseries['start_time'],
        feature_timeseries_blower,
        color='magenta',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='--',
        label='FAN Electricity',
    )

    timeseries = energy_timeseries[energy_timeseries['device_type'] == 'boiler']
    # fmt: off
    # pylint: disable=line-too-long
    if cumulative:
      feature_timeseries_gas = _to_kwh(timeseries['boiler_natural_gas_heating_energy_rate'])
      feature_timeseries_pump = _to_kwh(timeseries['boiler_pump_electrical_energy_rate'])
    else:
      feature_timeseries_gas = timeseries['boiler_natural_gas_heating_energy_rate'] / 1000.0
      feature_timeseries_pump = timeseries['boiler_pump_electrical_energy_rate'] / 1000.0
    # pylint: enable=line-too-long
    # fmt: on

    ax1.plot(
        timeseries['start_time'],
        feature_timeseries_gas,
        color='lime',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='-',
        label='BLR Gas',
    )

    ax1.plot(
        timeseries['start_time'],
        feature_timeseries_pump,
        color='lime',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='--',
        label='Pump Electricity',
    )

    if cumulative:
      label = 'HVAC Energy Consumption [kWh]'
    else:
      label = 'HVAC Power Consumption [kW]'

    self._format_plot(
        ax1,
        label,
        timeseries['start_time'].min(),
        timeseries['end_time'].max(),
        time_zone,
    )

  def _plot_energy_cost_timeline(
      self,
      ax1,
      reward_timeseries: pd.DataFrame,
      time_zone: str,
      cumulative: bool = False,
  ):
    """Plot energy cost timeline."""

    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    if cumulative:
      feature_timeseries_cost = reward_timeseries['electricity_energy_cost'].cumsum()  # pylint: disable=line-too-long
    else:
      feature_timeseries_cost = reward_timeseries['electricity_energy_cost']

    ax1.plot(
        local_times,
        feature_timeseries_cost,
        color='magenta',
        marker=None,
        alpha=1,
        lw=2,
        linestyle='-',
        label='Electricity',
    )

    self._format_plot(
        ax1,
        'Energy Cost [$]',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )

  def _plot_carbon_timeline(
      self, ax1, reward_timeseries, time_zone, cumulative=False
  ):
    """Plots carbon-emission timeline."""

    if cumulative:
      feature_timeseries_carbon = reward_timeseries['carbon_emitted'].cumsum()
    else:
      feature_timeseries_carbon = reward_timeseries['carbon_emitted']

    ax1.plot(
        reward_timeseries.index,
        feature_timeseries_carbon,
        color='white',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='-',
        label='Carbon',
    )

    self._format_plot(
        ax1,
        'Carbon emission [kg]',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )

  def _plot_occupancy_timeline(
      self, ax1, reward_timeseries: pd.DataFrame, time_zone: str
  ):
    """Plot occupancy timeline."""

    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    ax1.plot(
        local_times,
        reward_timeseries['occupancy'],
        color='cyan',
        marker=None,
        alpha=1,
        lw=2,
        linestyle='-',
        label='Num Occupants',
    )

    self._format_plot(
        ax1,
        'Occupancy',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )

  def _plot_temperature_timeline(
      self, ax1, zone_timeseries, outside_air_temperature_timeseries, time_zone
  ):
    """Plot temperature timeline."""

    zone_temps = pd.pivot_table(
        zone_timeseries,
        index=zone_timeseries['start_time'],
        columns='zone',
        values='zone_air_temperature',
    ).sort_index()

    zone_temps.quantile(q=0.25, axis=1)

    zone_temp_stats = pd.DataFrame({
        'min_temp': zone_temps.min(axis=1),
        'q25_temp': zone_temps.quantile(q=0.25, axis=1),
        'median_temp': zone_temps.median(axis=1),
        'q75_temp': zone_temps.quantile(q=0.75, axis=1),
        'max_temp': zone_temps.max(axis=1),
    })

    zone_heating_setpoints = (
        pd.pivot_table(
            zone_timeseries,
            index=zone_timeseries['start_time'],
            columns='zone',
            values='heating_setpoint_temperature',
        )
        .sort_index()
        .min(axis=1)
    )

    zone_cooling_setpoints = (
        pd.pivot_table(
            zone_timeseries,
            index=zone_timeseries['start_time'],
            columns='zone',
            values='cooling_setpoint_temperature',
        )
        .sort_index()
        .max(axis=1)
    )

    ax1.plot(
        zone_cooling_setpoints.index,
        zone_cooling_setpoints - self.KELVIN_TO_CELSIUS,
        color='yellow',
        lw=1,
    )

    ax1.plot(
        zone_cooling_setpoints.index,
        zone_heating_setpoints - self.KELVIN_TO_CELSIUS,
        color='yellow',
        lw=1,
    )

    ax1.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['min_temp'] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats['max_temp'] - self.KELVIN_TO_CELSIUS,
        facecolor='green',
        alpha=0.8,
    )

    ax1.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['q25_temp'] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats['q75_temp'] - self.KELVIN_TO_CELSIUS,
        facecolor='green',
        alpha=0.8,
    )

    ax1.plot(
        zone_temp_stats.index,
        zone_temp_stats['median_temp'] - self.KELVIN_TO_CELSIUS,
        color='white',
        lw=3,
        alpha=1.0,
    )

    ax1.plot(
        outside_air_temperature_timeseries.index,
        outside_air_temperature_timeseries - self.KELVIN_TO_CELSIUS,
        color='magenta',
        lw=3,
        alpha=1.0,
    )

    self._format_plot(
        ax1,
        'Temperature [C]',
        zone_temp_stats.index.min(),
        zone_temp_stats.index.max(),
        time_zone,
    )

  def _plot_action_timeline(
      self, ax1, action_timeseries, action_tuple, time_zone
  ):
    """Plots action timeline."""

    single_action_timeseries = action_timeseries[
        (action_timeseries['device_id'] == action_tuple[0])
        & (action_timeseries['setpoint_name'] == action_tuple[1])
    ]

    single_action_timeseries = single_action_timeseries.sort_values(by='timestamp')  # pylint: disable=line-too-long

    if action_tuple[1] in ['supply_water_setpoint', 'supply_air_heating_temperature_setpoint']:  # pylint: disable=line-too-long
      single_action_timeseries['setpoint_value'] = (
          single_action_timeseries['setpoint_value'] - self.KELVIN_TO_CELSIUS
      )

    ax1.plot(
        single_action_timeseries['timestamp'],
        single_action_timeseries['setpoint_value'],
        color='lime',
        marker=None,
        alpha=1,
        lw=4,
        linestyle='-',
        label=action_tuple[1],
    )

    self._format_plot(
        ax1,
        'Action',
        single_action_timeseries['timestamp'].min(),
        single_action_timeseries['timestamp'].max(),
        time_zone,
    )

  def _plot_timeseries_charts(self, reader, time_zone, step_count):
    """Plots timeseries charts and saves to file."""

    # fmt: off
    # pylint: disable=line-too-long
    observation_responses = reader.read_observation_responses(pd.Timestamp.min, pd.Timestamp.max)
    action_responses = reader.read_action_responses(pd.Timestamp.min, pd.Timestamp.max)
    reward_infos = reader.read_reward_infos(pd.Timestamp.min, pd.Timestamp.max)
    reward_responses = reader.read_reward_responses(pd.Timestamp.min, pd.Timestamp.max)
    # pylint: enable=line-too-long
    # fmt: on

    if not reward_infos or not reward_responses:
      logger.info('No reward data available for plotting')
      return

    action_timeseries = get_action_timeseries(action_responses)

    action_tuples = list(
        set([
            (row['device_id'], row['setpoint_name'])
            for _, row in action_timeseries.iterrows()
        ])
    )

    reward_timeseries = get_reward_timeseries(reward_infos, reward_responses, time_zone).sort_index()  # pylint: disable=line-too-long

    outside_air_temperature_timeseries = get_outside_air_temperature_timeseries(
        observation_responses, time_zone
    )

    zone_timeseries = get_zone_timeseries(reward_infos, time_zone)

    fig, axes = plt.subplots(
        nrows=6 + len(action_tuples),
        ncols=1,
        gridspec_kw={
            'height_ratios': [1, 1, 1, 1, 1, 1] + [1] * len(action_tuples)
        },
        squeeze=True,
    )
    fig.set_size_inches(24, 25)

    # fmt: off
    # pylint: disable=line-too-long
    energy_timeseries = get_energy_timeseries(reward_infos, time_zone)
    self._plot_reward_timeline(axes[0], reward_timeseries, time_zone)
    self._plot_energy_timeline(axes[1], energy_timeseries, time_zone, cumulative=True)
    self._plot_energy_cost_timeline(axes[2], reward_timeseries, time_zone, cumulative=True)
    self._plot_carbon_timeline(axes[3], reward_timeseries, time_zone, cumulative=True)
    self._plot_occupancy_timeline(axes[4], reward_timeseries, time_zone)
    self._plot_temperature_timeline(axes[5], zone_timeseries, outside_air_temperature_timeseries, time_zone)
    # pylint: enable=line-too-long
    # fmt: on

    for i, action_tuple in enumerate(action_tuples):
      self._plot_action_timeline(
          axes[6 + i], action_timeseries, action_tuple, time_zone
      )

    # Save figure instead of displaying
    fig_path = os.path.join(
        self._save_path, f'timeseries_step_{step_count}.png'
    )
    fig.savefig(fig_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    logger.info('Saved timeseries plot to %s', fig_path)

  def _render_env(self, env: environment.Environment, step_count: int):
    """Renders the environment and saves to file."""
    building_layout = env.building.simulator.building.floor_plan

    # Create a renderer
    renderer = building_renderer.BuildingRenderer(building_layout, 1)

    # Get the current temps to render
    temps = env.building.simulator.building.temp
    input_q = env.building.simulator.building.input_q

    # Render
    vmin = 285
    vmax = 305
    image = renderer.render(
        temps,
        cmap='bwr',
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        input_q=input_q,
        diff_range=0.5,
        diff_size=1,
    ).convert('RGB')

    # Save image instead of displaying
    timestamp = env.current_simulation_timestamp.strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(
        self._save_path, f'env_render_{step_count}_{timestamp}.png'
    )
    image.save(img_path)
    logger.info('Saved environment render to %s', img_path)

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Process a trajectory and render/plot if interval is reached.

    Args:
        trajectory: The trajectory to process.
    """
    logger.info('Called RenderingObserver observer...')

    reward = trajectory.reward
    self._cumulative_reward += reward
    self._counter += 1
    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._render_interval_steps == 0 and self._environment:
      logger.info('Rendering environment at step %d...', self._counter)
      execution_time = pd.Timestamp.now() - self._start_time
      mean_execution_time = execution_time.total_seconds() / self._counter

      logger.info(
          'Step %d: Cumulative reward = %.2f, Mean execution time = %.2fs',
          self._counter,
          float(self._cumulative_reward),
          mean_execution_time,
      )

      if self._environment.pyenv.envs[0].metrics_path is not None:
        logger.warning('Plotting timeseries charts...')
        reader = get_latest_episode_reader(self._environment.pyenv.envs[0].metrics_path)  # pylint: disable=line-too-long
        self._plot_timeseries_charts(reader, self._time_zone, self._counter)

      self._render_env(self._environment.pyenv.envs[0], self._counter)

  def reset(self) -> None:
    """Reset the observer to its initial state."""
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
