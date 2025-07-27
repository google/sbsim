---
layout: "default"
title: "Scripts"
nav_order: 7
parent: "Reinforcement Learning Module"
grand_parent: "Smart Control Project Documentation"
---

# Scripts

The `scripts` module provides command-line utilities for training, evaluating, and managing reinforcement learning experiments. These scripts streamline common RL tasks with standardized workflows.

## Training Script

`train.py` trains a reinforcement learning agent using a pre-populated replay buffer, managing agent creation, experience collection, and metrics logging.

### Parameters

| Parameter                              | Required | Default                          | Description                                                                 |
|----------------------------------------|----------|----------------------------------|-----------------------------------------------------------------------------|
| `--starter-buffer-path`                | Yes      | N/A                              | Path to the pre-populated replay buffer                                     |
| `--experiment-name`                    | Yes      | N/A                              | Name of the experiment for saving results                                   |
| `--agent-type`                         | No       | `'sac'`                          | Type of agent to train: `'sac'`, `'td3'`, or `'ddpg'`                       |
| `--train-iterations`                   | No       | `300`                            | Total number of training iterations                                         |
| `--collect-steps-per-training-iteration`| No       | `50`                             | Number of environment steps to collect per training iteration               |
| `--batch-size`                         | No       | `256`                            | Batch size for training (number of samples per gradient update)             |
| `--log-interval`                       | No       | `1`                              | Interval (in steps) for logging training metrics                            |
| `--eval-interval`                      | No       | `10`                             | Interval (in steps) for evaluating the agent                                |
| `--num-eval-episodes`                  | No       | `1`                              | Number of episodes to run during each evaluation                            |
| `--checkpoint-interval`                | No       | `10`                             | Interval (in steps) for checkpointing the replay buffer                     |
| `--learner-iterations`                 | No       | `200`                            | Number of gradient updates to perform per training iteration                |
| `--scenario-config-path`               | No       | `smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin` | Path to the scenario configuration file (.gin)                              |

### Example Usage

```bash
python scripts/train.py \
    --starter-buffer-path data/buffers/initial_buffer \
    --experiment-name hvac_control_sac \
    --agent-type sac \
    --train-iterations 300 \
    --collect-steps-per-training-iteration 50 \
    --batch-size 256 \
    --scenario-config-path configs/custom_config.gin
```

## Evaluation Script

`eval.py` evaluates a trained policy or the baseline schedule policy in a configured environment, producing performance metrics and optional trajectory data.

### Parameters

| Parameter             | Required | Default                                                      | Description                                                                 |
|-----------------------|----------|--------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--policy-dir`        | Yes      | N/A                                                          | Path to the saved policy directory or `"schedule"` for baseline policy      |
| `--gin-config`        | No       | `smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-7_starttimestamp-2023-07-06.gin` | Path to the environment configuration file (.gin)                           |
| `--num-eval-episodes` | No       | `1`                                                          | Number of episodes to run for evaluation                                    |
| `--experiment-name`   | Yes      | N/A                                                          | Name of the evaluation experiment for saving results                        |
| `--save-trajectory`   | No       | `True`                                                       | Whether to save detailed trajectory data for each episode                   |

### Example Usage

```bash
python scripts/eval.py \
    --policy-dir experiments/hvac_control_sac/policies/greedy_policy \
    --gin-config configs/building_sim.gin \
    --num-eval-episodes 5 \
    --experiment-name sac_evaluation \
    --save-trajectory False
```

## Buffer Population Script

`populate_starter_buffer.py` populates an initial replay buffer with exploration data using the baseline schedule policy, aiding off-policy learning.

### Parameters

| Parameter                   | Required | Default                                                      | Description                                                                 |
|-----------------------------|----------|--------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--buffer-name`             | Yes      | N/A                                                          | Name to identify the saved replay buffer                                    |
| `--capacity`                | No       | `50000`                                                      | Maximum capacity of the replay buffer                                       |
| `--steps-per-run`           | No       | `672`                                                        | Number of steps to collect per actor run (episode)                          |
| `--num-runs`                | No       | `10`                                                         | Number of actor runs (episodes) to perform                                  |
| `--sequence-length`         | No       | `2`                                                          | Sequence length for storing trajectories in the buffer                      |
| `--env-gin-config-file-path`| No       | `smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin` | Path to the environment configuration file (.gin)                           |

### Example Usage

```bash
python scripts/populate_starter_buffer.py \
    --buffer-name initial_exploration \
    --capacity 100000 \
    --steps-per-run 1000 \
    --num-runs 20 \
    --sequence-length 2 \
    --env-gin-config-file-path configs/custom_env.gin
```

## Configuration Generator Script

`generate_gin_config_files.py` generates multiple gin config files from a parameter grid for systematic experimentation.

### Parameters

| Parameter           | Required | Default                          | Description                                                                 |
|---------------------|----------|----------------------------------|-----------------------------------------------------------------------------|
| `base_config`       | Yes      | N/A                              | Path to the base gin config file (positional argument)                      |
| `--output-dir`      | No       | `'generated_configs'`            | Directory to save the generated config files                                |
| `--time-steps`      | No       | `'300'`                          | Comma-separated list of `time_step_sec` values to grid over                 |
| `--num-days`        | No       | `'1,7,14,30'`                   | Comma-separated list of `num_days_in_episode` values to grid over           |
| `--start-timestamps`| No       | `'2023-07-06'`                   | Comma-separated list of `start_timestamp` dates to grid over                |

### Example Usage

```bash
python scripts/generate_gin_config_files.py configs/base_config.gin \
    --output-dir configs/generated \
    --time-steps 300,600,900 \
    --num-days 1,7,14 \
    --start-timestamps 2023-07-06,2023-10-06
```

## Typical Workflow

A typical RL experiment workflow includes:

1. **Generate configurations:**

   ```bash
   python scripts/generate_gin_config_files.py configs/template.gin \
       --output-dir configs/generated
   ```

2. **Populate initial buffer:**

   ```bash
   python scripts/populate_starter_buffer.py \
       --buffer-name starter \
       --env-gin-config-file-path configs/generated/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin
   ```

3. **Train agent:**

   ```bash
   python scripts/train.py \
       --starter-buffer-path data/buffers/starter \
       --experiment-name my_experiment \
       --scenario-config-path configs/generated/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin
   ```

4. **Evaluate trained policy:**

   ```bash
   python scripts/eval.py \
       --policy-dir experiments/my_experiment/policies/greedy_policy \
       --gin-config configs/generated/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin \
       --experiment-name eval_my_experiment
   ```

5. **Compare against baseline:**

   ```bash
   python scripts/eval.py \
       --policy-dir schedule \
       --gin-config configs/generated/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin \
       --experiment-name baseline_evaluation
   ```

---

[Back to Reinforcement Learning](../reinforcement-learning-module.md)

[Back to Home](../../index.md)

---
