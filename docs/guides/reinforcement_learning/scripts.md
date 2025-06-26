# Reinforcement Learning Scripts

## Configuration Generation

```sh
python -m smart_control.reinforcement_learning.scripts.generate_gin_configs
```

```sh
python -m smart_control.reinforcement_learning.scripts.generate_gin_configs \
  /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/sim_config.gin \
  --output-dir /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs \
  --time-steps 900 \
  --num-days 14 \
  --start-timestamps 2023-07-21,2023-08-21,2023-10-21,2023-11-21 \
```

```sh
python scripts/generate_gin_config_files.py /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/sim_config.gin \
  --output-dir /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs \
  --time-steps 300,600,900 \
  --num-days 1,7,14 \
  --start-timestamps 2023-07-06,2023-08-06,2023-10-06
```

## Starter Buffer Population

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer
```

```sh
python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer \
    --buffer-name default-starter-buffer
```

## Training

```sh
python -m smart_control.reinforcement_learning.scripts.train \
    --starter-buffer-path path/to/the/starter/buffer
    --experiment-name my-experiment-1
```

```sh
python scripts/train.py \
    --starter-buffer-path data/starter_buffers/default_starter_buffer_seqlen2_exp6720/2025-04-04T06\:30\:49.808661634-04\:00/ \
    --experiment-name sac_multiple_episodes \
    --scenario-config-path "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-08-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin" \
    --eval-scenario-config-path "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-21.gin"
```

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
