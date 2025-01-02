import os
import json
import pandas as pd
import multiprocessing as mp
import argparse
from time import time
from tqdm import tqdm
from tf_agents.train.utils import spec_utils
from smart_control.mcts.mcts_utils import get_available_actions
from smart_control.mcts.execute_policy_utils import load_environment, compute_avg_return
from smart_control.mcts.SchedulePolicy import SchedulePolicy, FixedActionPolicy, ScheduleEvent, DeviceType
from smart_control.mcts.MonteCarloTreeSearch import SbSimMonteCarloTreeSearch, SbsimMonteCarloTreeSearchNode, NodeEnvironmentState


data_path = "/home/trigo/sbsim/sbsim/smart_control/configs/resources/sb1/"
default_env_config = os.path.join(data_path, "sim_config.gin")
default_env = load_environment(default_env_config)
default_action_sequence = [
    (DeviceType.HWS, 'supply_water_setpoint'),
    (DeviceType.AC, 'supply_air_heating_temperature_setpoint')
]


def get_policy_with_fixed_action(env, action):
    schedule_events = [
        ScheduleEvent(
            pd.Timedelta(0, unit='hour'),
            DeviceType.AC,
            'supply_air_heating_temperature_setpoint',
            action[0],
        ),
        ScheduleEvent(
            pd.Timedelta(0, unit='hour'),
            DeviceType.HWS,
            'supply_water_setpoint',
            action[1],
        )
    ]

    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)

    policy = FixedActionPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        action_sequence=default_action_sequence,
        schedule_events=schedule_events,
        action_normalizers=env._action_normalizers,
    )

    return policy


def get_default_schedule_policy(env):
    hod_cos_index = default_env._field_names.index('hod_cos_000')
    hod_sin_index = default_env._field_names.index('hod_sin_000')
    dow_cos_index = default_env._field_names.index('dow_cos_000')
    dow_sin_index = default_env._field_names.index('dow_sin_000')

    # Note that temperatures are specified in Kelvin:
    weekday_schedule_events = [
        ScheduleEvent(
            pd.Timedelta(6, unit='hour'),
            DeviceType.AC,
            'supply_air_heating_temperature_setpoint',
            292.0,
        ),
        ScheduleEvent(
            pd.Timedelta(19, unit='hour'),
            DeviceType.AC,
            'supply_air_heating_temperature_setpoint',
            285.0,
        ),
        ScheduleEvent(
            pd.Timedelta(6, unit='hour'),
            DeviceType.HWS,
            'supply_water_setpoint',
            350.0,
        ),
        ScheduleEvent(
            pd.Timedelta(19, unit='hour'),
            DeviceType.HWS,
            'supply_water_setpoint',
            315.0,
        ),
    ]

    weekend_holiday_schedule_events = [
        ScheduleEvent(
            pd.Timedelta(6, unit='hour'),
            DeviceType.AC,
            'supply_air_heating_temperature_setpoint',
            285.0,
        ),
        ScheduleEvent(
            pd.Timedelta(19, unit='hour'),
            DeviceType.AC,
            'supply_air_heating_temperature_setpoint',
            285.0,
        ),
        ScheduleEvent(
            pd.Timedelta(6, unit='hour'),
            DeviceType.HWS,
            'supply_water_setpoint',
            315.0,
        ),
        ScheduleEvent(
            pd.Timedelta(19, unit='hour'),
            DeviceType.HWS,
            'supply_water_setpoint',
            315.0,
        ),
    ]

    action_normalizers = env._action_normalizers

    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)
    local_start_time = env.current_simulation_timestamp.tz_convert(tz='US/Pacific')

    schedule_policy = SchedulePolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        action_sequence=default_action_sequence,
        weekday_schedule_events=weekday_schedule_events,
        weekend_holiday_schedule_events=weekend_holiday_schedule_events,
        dow_sin_index=dow_sin_index,
        dow_cos_index=dow_cos_index,
        hod_sin_index=hod_sin_index,
        hod_cos_index=hod_cos_index,
        local_start_time=local_start_time,
        action_normalizers=action_normalizers,
    )
    return schedule_policy


def main():

    start_time = time()

    # Parse experiment arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--expansion_num_steps", type=int, default=12)
    parser.add_argument("--rollout_num_steps", type=int, default=12)
    parser.add_argument("--num_processes", type=int, default=8)
    
    parser.add_argument("--t_water_low", type=int, default=285)
    parser.add_argument("--t_water_high", type=int, default=300)
    parser.add_argument("--t_air_low", type=int, default=310)
    parser.add_argument("--t_air_high", type=int, default=350)
    parser.add_argument("--t_water_step", type=int, default=5)
    parser.add_argument("--t_air_step", type=int, default=10)

    # if True, uses only the two actions from the default schedule
    parser.add_argument("--use_only_baseline_actions", type=bool, default=True)

    args = parser.parse_args()


    # Retrieve return information of default strategy
    return_by_timestamp = {}
    with open('./returns.json', 'r') as file:
        data = json.load(file)
    for timestamp, return_value in data:
        return_by_timestamp[pd.Timestamp(timestamp)] = return_value

    # Get all possible actions
    if args.use_only_baseline_actions:
        possible_actions = [(292, 350), (285, 315)]
    else:
        possible_actions = get_available_actions(args.t_water_low,
                                                args.t_water_high,
                                                args.t_air_low,
                                                args.t_air_high,
                                                args.t_water_step,
                                                args.t_air_step)

    # Create Monte Carlo Tree
    starting_node_environment_state = NodeEnvironmentState(
            node_temps=default_env.building._simulator._building.temp,
            node_timestamp=default_env.building._simulator._current_timestamp,
            node_state_return=0,
            node_previous_step=None)
    root = SbsimMonteCarloTreeSearchNode(starting_node_environment_state)
    tree = SbSimMonteCarloTreeSearch(root,
                                     return_by_timestamp,
                                     possible_actions,
                                     default_env_config,
                                     default_action_sequence,
                                     )
    
    mp.set_start_method("spawn") # required for multiprocessing to work
    progress_bar = tqdm(total=args.num_rollouts)

    while progress_bar.n < args.num_rollouts:

        nodes_for_expansion = tree.get_nodes_for_expansion(num_nodes=args.num_processes) # nodes for expansion is a list of tuples (node, action)
        expansion_work_items = [(node.node_environment_state, action, args.expansion_num_steps) for node, action in nodes_for_expansion]


        with mp.Pool(processes=args.num_processes) as pool:
            expansion_results = pool.map(expansion_worker, expansion_work_items)

        expansion_results = [(x, y, z) for (x, y), z in zip(nodes_for_expansion, expansion_results)]
        new_nodes = tree.perform_expansions(expansion_results)


        rollout_items = [(node.node_environment_state, return_by_timestamp, args.rollout_num_steps) for node in new_nodes]
        with mp.Pool(processes=args.num_processes) as pool:
            rollout_results = pool.map(rollout_worker, rollout_items)

        rollout_results = [(x, y, z) for x, (y, z) in zip(new_nodes, rollout_results)]
        tree.perform_backpropagations(rollout_results)

        progress_bar.update(len(nodes_for_expansion))
    
    end_time = time()
    
    experiment_results = {
        "args": {
            "num_rollouts": args.num_rollouts,
            "expansion_num_steps": args.expansion_num_steps,
            "rollout_num_steps": args.rollout_num_steps,
            "num_processes": args.num_processes,
            "t_water_low": args.t_water_low,
            "t_water_high": args.t_water_high,
            "t_air_low": args.t_air_low,
            "t_air_high": args.t_air_high,
            "t_water_step": args.t_water_step,
            "t_air_step": args.t_air_step
        },

        "tree_depth": tree.get_tree_depth(),
        "tree_size": tree.get_tree_size(),
        "returns_by_timestamp": tree.get_returns_by_timestamp(),
        "experiment_time": end_time - start_time
    }

    json.dump(experiment_results,
              open(f"experiment_results_two_actions/num_rollouts{ args.num_rollouts }num_processes{ args.num_processes }.json", "w"),
              indent=4)

    return


def expansion_worker(item):
    node_environment_state, action, expansion_steps = item
    env = SbsimMonteCarloTreeSearchNode.get_node_environment(default_env_config, node_environment_state)
    policy = get_policy_with_fixed_action(default_env, action)

    return SbsimMonteCarloTreeSearchNode.run_expansion(env, node_environment_state, policy, num_steps=expansion_steps)


def rollout_worker(item):
    node_environment_state, return_by_timestamp, rollout_steps = item
    env = SbsimMonteCarloTreeSearchNode.get_node_environment(default_env_config, node_environment_state)
    rollout_policy = get_default_schedule_policy(default_env)

    return SbsimMonteCarloTreeSearchNode.run_rollout(env,
                                                     node_environment_state,
                                                     rollout_policy,
                                                     return_by_timestamp,
                                                     rollout_steps=rollout_steps)


if __name__ == "__main__":
    main()
