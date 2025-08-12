# Reinforcement Learning Scripts

## Configuration Generation

By default, when training an RL agent, it will use configuration options defined
in the base gin config file (see
"smart_control/configs/resources/\<dataset_id>/sim_config.gin").

However if you would like to use different configuration options, you can use
the configuration generation script to flexibly create alternative config files
with slight modifications to the base config file.

Generate different configuration files to use during training:

```sh
python -m smart_control.reinforcement_learning.scripts.generate_gin_configs
```

By default, the script will use the following parameter grid:

- `time_steps`: `['300']`
- `num_days`: `['1', '7', '14', '30']`
- `start_timestamps`: ['2023-07-06']

Optionally pass any of these command line flags to customize the parameter grid:

```sh
python -m smart_control.reinforcement_learning.scripts.generate_gin_configs \
  --time_steps 300,600,900 \
  --num_days 1,7,14 \
  --start_timestamps 2023-07-06,2023-08-06,2023-10-06
```

This script will generate a different file for each combination of custom
parameter values you specify. The files will be written to the
"smart_control/configs/resources/\<dataset_id>/train_sim_configs/generated"
directory. Each file name will contain the parameter values you choose (e.g.
"step_300_days_1_start_20230706.gin").

## Starter Buffer Population

Populate an initial replay buffer with initial exploration data, to provide a
starting point when training RL agents:

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer
```

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer \
    --buffer_name buffer_xyz
    --config_path smart_control/configs/resources/sb1/sim_config.gin
```

This creates a directory corresponding with the buffer name in
"smart_control/reinforcement_learning/data/starter_buffers".

A "default" starter buffer has been created for example purposes:

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer \
    --buffer_name default \
    --num_runs 5 \
    --capacity 50000 \
    --steps_per_run 100 \
    --sequence_length 2
```

A "test" starter buffer has been created for testing purposes:

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer \
    --buffer_name test \
    --num_runs 1 \
    --steps_per_run 3 \
    --capacity 100 \
    --sequence_length 2
```

## RL Agent Training

Train a reinforcement learning agent.

```sh
python -m smart_control.reinforcement_learning.scripts.train \
    --experiment_name my-experiment-1
```

```sh
python -m smart_control.reinforcement_learning.scripts.train \
    --experiment_name my-experiment-1 \
    --agent_type="sac"
    --learner_iterations=3 \
    --train_iterations=10 \
    --collect_steps_per_training_iteration=5
```

This will generate a new experiment results directory under
"smart_control/reinforcement_learning/data/experiment_results/`experiment_name`".
In the experiment results directory will be the following files and directories:

- "collect" directory
- "eval" directory
- "metrics" directory
- "replay_buffer" directory
- "experiment_parameters.json" file
- "experiment_parameters.txt" file

## Evaluation

```sh
python -m smart_control.reinforcement_learning.scripts.eval
```

```sh
python scripts/eval.py
  --policy-dir experiment_results/ddpg_train_run-july-6th_2025_04_07-12:50:40/policies/
  --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin
  --experiment-name ddpg_train-summer_eval-winter
```
