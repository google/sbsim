# @title Imports
import tensorflow as tf
import datetime
import pytz
import enum
import functools
import os
import os
import time
import gin
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import mediapy as media
import reverb
import pandas as pd
import numpy as np
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator
from matplotlib import patches
from absl import logging
from dataclasses import dataclass
from typing import Final, Sequence
from typing import Optional
from typing import Union, cast
from tf_agents.specs import tensor_spec
from tf_agents.typing import types
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step
from tf_agents.train.utils import train_utils
from tf_agents.train.utils import spec_utils
from tf_agents.train import triggers
from tf_agents.train import learner
from tf_agents.train import actor
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.policies import tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import greedy_policy
from tf_agents.networks import sequential
from tf_agents.networks import nest_map
from tf_agents.metrics import py_metrics
from tf_agents.keras_layers import inner_reshape
from tf_agents.drivers import py_driver
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.sac import sac_agent
from smart_control.utils import environment_utils
from smart_control.utils import histogram_reducer
from smart_control.utils import writer_lib
from smart_control.utils import reader_lib
from smart_control.utils import observation_normalizer
from smart_control.utils import conversion_utils
from smart_control.utils import controller_writer
from smart_control.utils import controller_reader
from smart_control.utils import building_renderer
from smart_control.utils import bounded_action_normalizer
from smart_control.simulator import stochastic_convection_simulator
from smart_control.simulator import step_function_occupancy
from smart_control.simulator import simulator_building
from smart_control.simulator import rejection_simulator_building
from smart_control.simulator import randomized_arrival_departure_occupancy
from smart_control.reward import setpoint_energy_carbon_reward
from smart_control.reward import setpoint_energy_carbon_regret
from smart_control.reward import natural_gas_energy_cost
from smart_control.reward import electricity_energy_cost
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_building_pb2
from smart_control.environment.environment import Environment
from smart_control.environment import environment


os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'


data_path = "/home/trigo/sbsim/sbsim/smart_control/configs/resources/sb1/"
metrics_path = "/home/trigo/sbsim/sbsim/metrics"  # @param {type:"string"}
output_data_path = '/home/trigo/sbsim/sbsim/output'  # @param {type:"string"}
root_dir = "/home/trigo/sbsim/sbsim/root"  # @param {type:"string"}


# @title Plotting Utities
reward_shift = 0
reward_scale = 1.0
person_productivity_hour = 300.0
KELVIN_TO_CELSIUS = 273.15


# @title Plotting Utities
reward_shift = 0
reward_scale = 1.0
person_productivity_hour = 300.0


KELVIN_TO_CELSIUS = 273.15


def logging_info(*args):
    logging.info(*args)
    print(*args)


def render_env(env: environment.Environment):
    """Renders the environment."""
    building_layout = env.building._simulator._building._floor_plan

    # create a renderer
    renderer = building_renderer.BuildingRenderer(building_layout, 1)

    # get the current temps to render
    # this also is not ideal, since the temps are not fully exposed.
    # V Ideally this should be a publicly accessable field
    temps = env.building._simulator._building.temp

    input_q = env.building._simulator._building.input_q

    # render
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
    media.show_image(
        image, title='Environment %s' % env.current_simulation_timestamp
    )


def get_energy_timeseries(reward_infos, time_zone: str) -> pd.DataFrame:
    """Returns a timeseries of energy rates."""

    start_times = []
    end_times = []

    device_ids = []
    device_types = []
    air_handler_blower_electrical_energy_rates = []
    air_handler_air_conditioner_energy_rates = []
    boiler_natural_gas_heating_energy_rates = []
    boiler_pump_electrical_energy_rates = []

    for reward_info in reward_infos:
        end_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_info.end_timestamp
        ).tz_convert(time_zone)
        start_timestamp = end_timestamp - pd.Timedelta(300, unit='second')

        for air_handler_id in reward_info.air_handler_reward_infos:
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            device_ids.append(air_handler_id)
            device_types.append('air_handler')

            air_handler_blower_electrical_energy_rates.append(
                reward_info.air_handler_reward_infos[
                    air_handler_id
                ].blower_electrical_energy_rate
            )
            air_handler_air_conditioner_energy_rates.append(
                reward_info.air_handler_reward_infos[
                    air_handler_id
                ].air_conditioning_electrical_energy_rate
            )
            boiler_natural_gas_heating_energy_rates.append(0)
            boiler_pump_electrical_energy_rates.append(0)

        for boiler_id in reward_info.boiler_reward_infos:
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            device_ids.append(boiler_id)
            device_types.append('boiler')

            air_handler_blower_electrical_energy_rates.append(0)
            air_handler_air_conditioner_energy_rates.append(0)

            boiler_natural_gas_heating_energy_rates.append(
                reward_info.boiler_reward_infos[
                    boiler_id
                ].natural_gas_heating_energy_rate
            )
            boiler_pump_electrical_energy_rates.append(
                reward_info.boiler_reward_infos[boiler_id].pump_electrical_energy_rate
            )

    df_map = {
        'start_time': start_times,
        'end_time': end_times,
        'device_id': device_ids,
        'device_type': device_types,
        'air_handler_blower_electrical_energy_rate': (
            air_handler_blower_electrical_energy_rates
        ),
        'air_handler_air_conditioner_energy_rate': (
            air_handler_air_conditioner_energy_rates
        ),
        'boiler_natural_gas_heating_energy_rate': (
            boiler_natural_gas_heating_energy_rates
        ),
        'boiler_pump_electrical_energy_rate': boiler_pump_electrical_energy_rates,
    }
    df = pd.DataFrame(df_map).sort_values('start_time')
    return df


def get_outside_air_temperature_timeseries(
    observation_responses,
    time_zone: str,
) -> pd.Series:
    """Returns a timeseries of outside air temperature."""
    temps = []
    for i in range(len(observation_responses)):
        temp = [
            (
                conversion_utils.proto_to_pandas_timestamp(
                    sor.timestamp
                ).tz_convert(time_zone)
                - pd.Timedelta(300, unit='second'),
                sor.continuous_value,
            )
            for sor in observation_responses[i].single_observation_responses
            if sor.single_observation_request.measurement_name
            == 'outside_air_temperature_sensor'
        ][0]
        temps.append(temp)

    res = list(zip(*temps))
    return pd.Series(res[1], index=res[0]).sort_index()


def get_reward_timeseries(
    reward_infos,
    reward_responses,
    time_zone: str,
) -> pd.DataFrame:
    """Returns a timeseries of reward values."""
    cols = [
        'agent_reward_value',
        'electricity_energy_cost',
        'carbon_emitted',
        'occupancy',
    ]
    df = pd.DataFrame(columns=cols)

    for i in range(min(len(reward_responses), len(reward_infos))):
        step_start_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_infos[i].start_timestamp
        ).tz_convert(time_zone)
        step_end_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_infos[i].end_timestamp
        ).tz_convert(time_zone)
        delta_time_sec = (step_end_timestamp -
                          step_start_timestamp).total_seconds()
        occupancy = np.sum([
            reward_infos[i].zone_reward_infos[zone_id].average_occupancy
            for zone_id in reward_infos[i].zone_reward_infos
        ])

        df.loc[
            conversion_utils.proto_to_pandas_timestamp(
                reward_infos[i].start_timestamp
            ).tz_convert(time_zone)
        ] = [
            reward_responses[i].agent_reward_value,
            reward_responses[i].electricity_energy_cost,
            reward_responses[i].carbon_emitted,
            occupancy,
        ]

    df = df.sort_index()
    df['cumulative_reward'] = df['agent_reward_value'].cumsum()
    logging_info('Cumulative reward: %4.2f' % df.iloc[-1]['cumulative_reward'])
    return df


def format_plot(
    ax1, xlabel: str, start_time: int, end_time: int, time_zone: str
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


def plot_occupancy_timeline(
    ax1, reward_timeseries: pd.DataFrame, time_zone: str
):
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
    format_plot(
        ax1,
        'Occupancy',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )


def plot_energy_cost_timeline(
    ax1,
    reward_timeseries: pd.DataFrame,
    time_zone: str,
    cumulative: bool = False,
):
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]
    if cumulative:
        feature_timeseries_cost = reward_timeseries[
            'electricity_energy_cost'
        ].cumsum()
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

    format_plot(
        ax1,
        'Energy Cost [$]',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )


def plot_reward_timeline(ax1, reward_timeseries, time_zone):

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
    format_plot(
        ax1,
        'Agent Reward',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )


def plot_energy_timeline(ax1, energy_timeseries, time_zone, cumulative=False):

    def _to_kwh(
        energy_rate: float,
        step_interval: pd.Timedelta = pd.Timedelta(5, unit='minute'),
    ) -> float:
        kw_power = energy_rate / 1000.0
        hwh_power = kw_power * step_interval / pd.Timedelta(1, unit='hour')
        return hwh_power.cumsum()

    timeseries = energy_timeseries[
        energy_timeseries['device_type'] == 'air_handler'
    ]

    if cumulative:
        feature_timeseries_ac = _to_kwh(
            timeseries['air_handler_air_conditioner_energy_rate']
        )
        feature_timeseries_blower = _to_kwh(
            timeseries['air_handler_blower_electrical_energy_rate']
        )
    else:
        feature_timeseries_ac = (
            timeseries['air_handler_air_conditioner_energy_rate'] / 1000.0
        )
        feature_timeseries_blower = (
            timeseries['air_handler_blower_electrical_energy_rate'] / 1000.0
        )

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
    if cumulative:
        feature_timeseries_gas = _to_kwh(
            timeseries['boiler_natural_gas_heating_energy_rate']
        )
        feature_timeseries_pump = _to_kwh(
            timeseries['boiler_pump_electrical_energy_rate']
        )
    else:
        feature_timeseries_gas = (
            timeseries['boiler_natural_gas_heating_energy_rate'] / 1000.0
        )
        feature_timeseries_pump = (
            timeseries['boiler_pump_electrical_energy_rate'] / 1000.0
        )

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

    format_plot(
        ax1,
        label,
        timeseries['start_time'].min(),
        timeseries['end_time'].max(),
        time_zone,
    )


def plot_carbon_timeline(ax1, reward_timeseries, time_zone, cumulative=False):
    """Plots carbon-emission timeline."""

    if cumulative:
        feature_timeseries_carbon = reward_timeseries['carbon_emitted'].cumsum(
        )
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
    format_plot(
        ax1,
        'Carbon emission [kg]',
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone,
    )


def get_zone_timeseries(reward_infos, time_zone):
    """Converts reward infos to a timeseries dataframe."""

    start_times = []
    end_times = []
    zones = []
    heating_setpoints = []
    cooling_setpoints = []
    zone_air_temperatures = []
    air_flow_rate_setpoints = []
    air_flow_rates = []
    average_occupancies = []

    for reward_info in reward_infos:
        start_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_info.end_timestamp
        ).tz_convert(time_zone) - pd.Timedelta(300, unit='second')
        end_timestamp = conversion_utils.proto_to_pandas_timestamp(
            reward_info.end_timestamp
        ).tz_convert(time_zone)

        for zone_id in reward_info.zone_reward_infos:
            zones.append(zone_id)
            start_times.append(start_timestamp)
            end_times.append(end_timestamp)

            heating_setpoints.append(
                reward_info.zone_reward_infos[zone_id].heating_setpoint_temperature
            )
            cooling_setpoints.append(
                reward_info.zone_reward_infos[zone_id].cooling_setpoint_temperature
            )

            zone_air_temperatures.append(
                reward_info.zone_reward_infos[zone_id].zone_air_temperature
            )
            air_flow_rate_setpoints.append(
                reward_info.zone_reward_infos[zone_id].air_flow_rate_setpoint
            )
            air_flow_rates.append(
                reward_info.zone_reward_infos[zone_id].air_flow_rate
            )
            average_occupancies.append(
                reward_info.zone_reward_infos[zone_id].average_occupancy
            )

    df_map = {
        'start_time': start_times,
        'end_time': end_times,
        'zone': zones,
        'heating_setpoint_temperature': heating_setpoints,
        'cooling_setpoint_temperature': cooling_setpoints,
        'zone_air_temperature': zone_air_temperatures,
        'air_flow_rate_setpoint': air_flow_rate_setpoints,
        'air_flow_rate': air_flow_rates,
        'average_occupancy': average_occupancies,
    }
    return pd.DataFrame(df_map).sort_values('start_time')


def get_action_timeseries(action_responses):
    """Converts action responses to a dataframe."""
    timestamps = []
    device_ids = []
    setpoint_names = []
    setpoint_values = []
    response_types = []
    for action_response in action_responses:

        timestamp = conversion_utils.proto_to_pandas_timestamp(
            action_response.timestamp
        )
        for single_action_response in action_response.single_action_responses:
            device_id = single_action_response.request.device_id
            setpoint_name = single_action_response.request.setpoint_name
            setpoint_value = single_action_response.request.continuous_value
            response_type = single_action_response.response_type

            timestamps.append(timestamp)
            device_ids.append(device_id)
            setpoint_names.append(setpoint_name)
            setpoint_values.append(setpoint_value)
            response_types.append(response_type)

    return pd.DataFrame({
        'timestamp': timestamps,
        'device_id': device_ids,
        'setpoint_name': setpoint_names,
        'setpoint_value': setpoint_values,
        'response_type': response_types,
    })


def plot_action_timeline(ax1, action_timeseries, action_tuple, time_zone):
    """Plots action timeline."""

    single_action_timeseries = action_timeseries[
        (action_timeseries['device_id'] == action_tuple[0])
        & (action_timeseries['setpoint_name'] == action_tuple[1])
    ]
    single_action_timeseries = single_action_timeseries.sort_values(
        by='timestamp'
    )

    if action_tuple[1] in [
        'supply_water_setpoint',
        'supply_air_heating_temperature_setpoint',
    ]:
        single_action_timeseries['setpoint_value'] = (
            single_action_timeseries['setpoint_value'] - KELVIN_TO_CELSIUS
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
    title = '%s %s' % (action_tuple[0], action_tuple[1])
    format_plot(
        ax1,
        'Action',
        single_action_timeseries['timestamp'].min(),
        single_action_timeseries['timestamp'].max(),
        time_zone,
    )


def get_outside_air_temperature_timeseries(observation_responses, time_zone):
    temps = []
    for i in range(len(observation_responses)):
        temp = [
            (
                conversion_utils.proto_to_pandas_timestamp(
                    sor.timestamp
                ).tz_convert(time_zone),
                sor.continuous_value,
            )
            for sor in observation_responses[i].single_observation_responses
            if sor.single_observation_request.measurement_name
            == 'outside_air_temperature_sensor'
        ][0]
        temps.append(temp)

    res = list(zip(*temps))
    return pd.Series(res[1], index=res[0]).sort_index()


def plot_temperature_timeline(
    ax1, zone_timeseries, outside_air_temperature_timeseries, time_zone
):
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
        zone_cooling_setpoints - KELVIN_TO_CELSIUS,
        color='yellow',
        lw=1,
    )
    ax1.plot(
        zone_cooling_setpoints.index,
        zone_heating_setpoints - KELVIN_TO_CELSIUS,
        color='yellow',
        lw=1,
    )

    ax1.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['min_temp'] - KELVIN_TO_CELSIUS,
        zone_temp_stats['max_temp'] - KELVIN_TO_CELSIUS,
        facecolor='green',
        alpha=0.8,
    )
    ax1.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['q25_temp'] - KELVIN_TO_CELSIUS,
        zone_temp_stats['q75_temp'] - KELVIN_TO_CELSIUS,
        facecolor='green',
        alpha=0.8,
    )
    ax1.plot(
        zone_temp_stats.index,
        zone_temp_stats['median_temp'] - KELVIN_TO_CELSIUS,
        color='white',
        lw=3,
        alpha=1.0,
    )
    ax1.plot(
        outside_air_temperature_timeseries.index,
        outside_air_temperature_timeseries - KELVIN_TO_CELSIUS,
        color='magenta',
        lw=3,
        alpha=1.0,
    )
    format_plot(
        ax1,
        'Temperature [C]',
        zone_temp_stats.index.min(),
        zone_temp_stats.index.max(),
        time_zone,
    )


def plot_timeseries_charts(reader, time_zone):
    """Plots timeseries charts."""

    observation_responses = reader.read_observation_responses(
        pd.Timestamp.min, pd.Timestamp.max
    )
    action_responses = reader.read_action_responses(
        pd.Timestamp.min, pd.Timestamp.max
    )
    reward_infos = reader.read_reward_infos(pd.Timestamp.min, pd.Timestamp.max)
    reward_responses = reader.read_reward_responses(
        pd.Timestamp.min, pd.Timestamp.max
    )

    if len(reward_infos) == 0 or len(reward_responses) == 0:
        return

    action_timeseries = get_action_timeseries(action_responses)
    action_tuples = list(
        set([
            (row['device_id'], row['setpoint_name'])
            for _, row in action_timeseries.iterrows()
        ])
    )

    reward_timeseries = get_reward_timeseries(
        reward_infos, reward_responses, time_zone
    ).sort_index()
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

    energy_timeseries = get_energy_timeseries(reward_infos, time_zone)
    plot_reward_timeline(axes[0], reward_timeseries, time_zone)
    plot_energy_timeline(axes[1], energy_timeseries,
                         time_zone, cumulative=True)
    plot_energy_cost_timeline(
        axes[2], reward_timeseries, time_zone, cumulative=True
    )
    plot_carbon_timeline(axes[3], reward_timeseries,
                         time_zone, cumulative=True)
    plot_occupancy_timeline(axes[4], reward_timeseries, time_zone)
    plot_temperature_timeline(
        axes[5], zone_timeseries, outside_air_temperature_timeseries, time_zone
    )

    for i, action_tuple in enumerate(action_tuples):
        plot_action_timeline(
            axes[6 + i], action_timeseries, action_tuple, time_zone
        )

    plt.show()


def remap_filepath(filepath) -> str:
    return filepath


def load_environment(gin_config_file: str):
    """Returns an Environment from a config file."""
    # Global definition is required by Gin library to instantiate Environment.
    # global environment  # pylint: disable=global-variable-not-assigned

    with gin.unlock_config():
        gin.clear_config()
        gin.parse_config_file(gin_config_file)
        return Environment()  # pylint: disable=no-value-for-parameter


def get_latest_episode_reader(
    metrics_path: str,
) -> controller_reader.ProtoReader:

    episode_infos = controller_reader.get_episode_data(
        metrics_path).sort_index()
    selected_episode = episode_infos.index[-1]
    episode_path = os.path.join(metrics_path, selected_episode)
    reader = controller_reader.ProtoReader(episode_path)
    return reader


@gin.configurable
def get_histogram_path():
    return data_path


@gin.configurable
def get_reset_temp_values():
    reset_temps_filepath = remap_filepath(
        os.path.join(data_path, "reset_temps.npy")
    )

    return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path():
    return remap_filepath(
        os.path.join(data_path, "double_resolution_zone_1_2.npy")
    )


@gin.configurable
def get_metrics_path():
    return os.path.join(metrics_path, "metrics")


@gin.configurable
def get_weather_path():
    return remap_filepath(
        os.path.join(
            data_path, "local_weather_moffett_field_20230701_20231122.csv"
        )
    )


# @title Load the environments
histogram_parameters_tuples = (
    ('zone_air_temperature_sensor', (285., 286., 287., 288, 289., 290., 291.,
     292., 293., 294., 295., 296., 297., 298., 299., 300., 301, 302, 303)),
    ('supply_air_damper_percentage_command', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    ('supply_air_flowrate_setpoint', (0., 0.05, .1, .2, .3, .4, .5,  .7,  .9)),
)


@gin.configurable
def get_histogram_reducer():

    reader = controller_reader.ProtoReader(data_path)

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=histogram_parameters_tuples,
        reader=reader,
        normalize_reduce=True,
    )
    return hr


# @title Define a method to execute the policy on the environment.
def get_trajectory(time_step, current_action: policy_step.PolicyStep):
    """Get the trajectory for the current action and time step."""
    observation = time_step.observation
    action = current_action.action
    policy_info = ()
    reward = time_step.reward
    discount = time_step.discount

    if time_step.is_first():
        traj = trajectory.first(observation, action,
                                policy_info, reward, discount)

    elif time_step.is_last():
        traj = trajectory.last(observation, action,
                               policy_info, reward, discount)

    else:
        traj = trajectory.mid(observation, action,
                              policy_info, reward, discount)
    return traj


def compute_avg_return(
    environment,
    policy,
    num_episodes=1,
    time_zone: str = "US/Pacific",
    render_interval_steps: int = 24,
    trajectory_observers=None,
    num_steps=6
):
    """Computes the average return of the policy on the environment.

    Args:
      environment: environment.Environment
      policy: policy.Policy
      num_episodes: total number of eposides to run.
      time_zone: time zone of the environment
      render_interval_steps: Number of steps to take between rendering.
      trajectory_observers: list of trajectory observers for use in rendering.
    """

    total_return = 0.0
    return_by_simtime = []
    for _ in range(num_episodes):

        time_step = environment.current_time_step()
        if not time_step:
            time_step = environment.reset()

        episode_return = 0.0
        t0 = time.time()
        epoch = t0

        step_id = 0
        execution_times = []

        for _ in range(num_steps):

            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)

            if trajectory_observers is not None:
                traj = get_trajectory(time_step, action_step)
                for observer in trajectory_observers:
                    observer(traj)

            episode_return += time_step.reward
            t1 = time.time()
            dt = t1 - t0
            episode_seconds = t1 - epoch
            execution_times.append(dt)
            sim_time = environment.current_simulation_timestamp.tz_convert(
                time_zone)

            return_by_simtime.append([sim_time, episode_return])

            print(
                "Step %5d Sim Time: %s, Reward: %8.2f, Return: %8.2f, Mean Step Time:"
                " %8.2f s, Episode Time: %8.2f s"
                % (
                    step_id,
                    sim_time.strftime("%Y-%m-%d %H:%M"),
                    time_step.reward,
                    episode_return,
                    np.mean(execution_times),
                    episode_seconds,
                )
            )

            if (step_id > 0) and (step_id % render_interval_steps == 0):
                if environment._metrics_path:
                    clear_output(wait=True)
                    reader = get_latest_episode_reader(
                        environment._metrics_path)
                    plot_timeseries_charts(reader, time_zone)
                render_env(environment)

            t0 = t1
            step_id += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return, return_by_simtime
