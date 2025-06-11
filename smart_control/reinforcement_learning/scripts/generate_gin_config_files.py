#!/usr/bin/env python3
"""
Grid Configuration Generator for Gin Config Files

This script generates multiple variations of a gin config file by creating a grid
of different values for specified parameters.
"""

import argparse
import logging
import os
import re
from itertools import product

from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.utils.constants import ROOT_DIR

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
)


def read_config_file(filepath):
  """Read the base configuration file."""
  with open(filepath, 'r') as f:
    return f.read()


def modify_config(config_content, param_name, param_value):
  """
  Modify a specific parameter in the config content.
  Matches parameter assignments with literal values (numbers or quoted strings)
  but not function calls that start with @ or contain parentheses.
  Returns the modified config content.
  """
  # This pattern has several components:
  # 1. Match line start or after newline
  # 2. Capture any leading text
  # 3. Capture the parameter name, equals sign, and surrounding whitespace
  # 4. Capture the value, which can be either:
  #    - A quoted string (with ' or ")
  #    - Or a sequence that doesn't start with @ and doesn't contain ()
  # 5. Capture the end of line

  pattern = rf'(^|\n)(.*?)({re.escape(param_name)}\s*=\s*)((?:[\'\"].*?[\'\"])|(?:[^@\n][^()\n]*))($|\n)'
  # Format replacement to preserve surrounding context
  replacement = r'\g<1>\g<2>\g<3>{}\g<5>'.format(param_value)

  modified_content = re.sub(
      pattern, replacement, config_content, flags=re.MULTILINE
  )

  if modified_content == config_content:
    logger.warning(
        f"Warning: Parameter '{param_name}' not found in config file."
    )

  return modified_content


def generate_configs(base_config_path, output_dir, param_grids):
  """
  Generate multiple config files based on parameter grids.

  Args:
      base_config_path: Path to the base config file
      output_dir: Directory to save generated config files
      param_grids: Dictionary mapping parameter names to lists of values
  """
  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Read the base config file
  base_config = read_config_file(base_config_path)

  # Get parameter names and their possible values
  param_names = list(param_grids.keys())
  param_values = [param_grids[name] for name in param_names]

  # Generate all combinations of parameter values
  for combination in product(*param_values):
    # Create a new config file for each combination
    modified_config = base_config

    # Build filename parts and track modifications for this combination
    filename_parts = []

    for i, param_name in enumerate(param_names):
      param_value = combination[i]
      modified_config = modify_config(modified_config, param_name, param_value)

      # Add to filename parts (clean parameter name and value)
      clean_name = param_name.replace('_', '')

      if param_name == 'start_timestamp':
        filename_parts.append(f'{clean_name}-{param_value[1:11]}')
      else:
        filename_parts.append(f'{clean_name}-{param_value}')

    # Generate a filename based on the parameter values
    output_filename = f"config_{'_'.join(filename_parts)}.gin"
    output_path = os.path.join(output_dir, output_filename)

    # Write the modified config to a new file
    with open(output_path, 'w') as f:
      f.write(modified_config)

    logger.info(f'Generated: {output_path}')


def main():
  parser = argparse.ArgumentParser(
      description='Generate grid of gin config files'
  )
  parser.add_argument(
      'base_config',
      default=os.path.join(
          ROOT_DIR,
          'smart_control',
          'configs',
          'resources',
          'sb1',
          'sim_config.gin',
      ),
      help='Path to the base gin config file',
  )
  parser.add_argument(
      '--output-dir',
      default=os.path.join(CONFIG_PATH, 'generated_configs'),
      help='Directory to save generated config files',
  )
  parser.add_argument(
      '--time-steps',
      type=str,
      default='300',
      help='Comma-separated list of time_step_sec values',
  )
  parser.add_argument(
      '--num-days',
      type=str,
      default='1,7,14,30',
      help='Comma-separated list of num_days_in_episode values',
  )
  parser.add_argument(
      '--start-timestamps',
      type=str,
      default='2023-07-06',
      help='Comma-separated list of start_timestamp dates',
  )

  args = parser.parse_args()

  # This ensures that it works both with absolute and relative paths
  if not os.path.isabs(args.base_config):
    args.base_config = os.path.join(ROOT_DIR, args.base_config)
  if not os.path.isabs(args.output_dir):
    args.output_dir = os.path.join(ROOT_DIR, args.output_dir)

  # Convert comma-separated values to lists
  time_steps = [step.strip() for step in args.time_steps.split(',')]
  num_days = [days.strip() for days in args.num_days.split(',')]
  start_timestamps = [
      f"'{ timestamp.strip() } 07:00:00+00:00'"
      for timestamp in args.start_timestamps.split(',')
  ]

  logger.info(start_timestamps)

  # Define the parameter grid
  param_grid = {
      'time_step_sec': time_steps,
      'num_days_in_episode': num_days,
      'start_timestamp': start_timestamps,
  }

  # Generate configurations
  generate_configs(args.base_config, args.output_dir, param_grid)

  logger.info(
      f'Generated {len(time_steps) * len(num_days)} configuration files in'
      f' {args.output_dir}'
  )


if __name__ == '__main__':
  main()
