#!/bin/bash

# Run the generator script
python scripts/generate_gin_config_files.py /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/sim_config.gin \
  --output-dir /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs \
  --time-steps 300,600,900 \
  --num-days 1,7,14 \
  --start-timestamps 2023-07-06,2023-08-06,2023-10-06
