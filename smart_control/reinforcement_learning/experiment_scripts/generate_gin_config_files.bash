#!/bin/bash

# Run the generator script
python scripts/generate_gin_config_files.py /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/sim_config.gin \
  --output-dir /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs \
  --time-steps 900 \
  --num-days 14 \
  --start-timestamps 2023-07-21,2023-08-21,2023-10-21,2023-11-21 \
