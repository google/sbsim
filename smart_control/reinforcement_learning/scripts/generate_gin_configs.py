"""
Grid Configuration Generator for Gin Config Files

This script generates multiple variations of a gin config file by creating a
grid of different values for specified parameters.
"""

from itertools import product
import logging
import os
import re
from typing import Sequence

from absl import app
from absl import flags

from smart_control.utils.constants import ROOT_DIR
from smart_control.utils.constants import SB1_GIN_CONFIG_FILEPATH
from smart_control.utils.constants import SB1_TRAIN_CONFIGS_DIR

SB1_GENERATED_CONFIGS_DIR = os.path.join(SB1_TRAIN_CONFIGS_DIR, 'generated')

# LOGGING

logger = logging.getLogger(__name__)

# logging.basicConfig(
#    level=logging.WARNING,
#    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
# )

logging.basicConfig(
    level=logging.INFO,
    format='[%(message)s]',
)

# FLAGS

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='base_config',
    default=SB1_GIN_CONFIG_FILEPATH,
    help='Path to the base gin config file',
)
flags.DEFINE_string(
    name='output_dir',
    default=SB1_GENERATED_CONFIGS_DIR,
    help='Directory to save generated config files',
)
flags.DEFINE_list(
    name='time_steps',
    default=['300'],
    help='Comma-separated list of time_step_sec values',
)
flags.DEFINE_list(
    name='num_days',
    default=['1', '7', '14', '30'],
    help='Comma-separated list of num_days_in_episode values',
)
flags.DEFINE_list(
    name='start_timestamps',
    default=['2023-07-06'],
    help='Comma-separated list of start_timestamp dates',
)

# FUNCTIONS


def read_config_file(filepath):
  """Read the base configuration file."""
  with open(filepath, 'r', encoding='utf-8') as f:
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

  pattern = (
      rf'(^|\n)'
      rf'(.*?)'
      rf'({re.escape(param_name)}\s*=)'
      rf'((?:[\'\"].*?[\'\"])|(?:[^@\n][^()\n]*))'
      rf'($|\n)'
  )
  # Format replacement to preserve surrounding context
  replacement = rf'\g<1>\g<2>\g<3>{param_value}\g<5>'

  modified_content = re.sub(
      pattern, replacement, config_content, flags=re.MULTILINE
  )

  if modified_content == config_content:
    logger.warning(
        "Warning: Parameter '%s' not found in config file.", param_name
    )

  return modified_content


# def generate_config(base_config_path: str, output_dir: str, params: dict):
#  pass


def generate_configs(
    params_grid: dict,
    base_config_path: str = SB1_GIN_CONFIG_FILEPATH,
    output_dir: str = SB1_GENERATED_CONFIGS_DIR,
):
  """
  Generate multiple config files based on parameter grids.

  Args:
      base_config_path: Path to the base config file
      output_dir: Directory to save generated config files
      params_grid: Dictionary mapping parameter names to lists of values

  Example:
    ```py
    grid = {
        'time_step_sec': ['300'],
        'num_days_in_episode': ['1', '7', '14', '30'],
        'start_timestamp': ['2023-07-06 07:00:00+00:00']
    }
    generate_configs("/path/to/my_config.gin", "/path/to/output/dir", grid)
    ```
  """
  os.makedirs(output_dir, exist_ok=True)

  base_config = read_config_file(base_config_path)

  param_names = list(params_grid.keys())
  param_values = list(params_grid.values())

  param_filename_aliases = {
      'time_step_sec': 'step',
      'num_days_in_episode': 'days',
      'start_timestamp': 'start',
      # can add more filename aliases here
  }

  # Generate all combinations of parameter values
  for combination in product(*param_values):
    # combination is like ('300', '1', '2023-07-06 07:00:00+00:00')

    # params = dict(zip(param_names, combination))
    # > {'time_step_sec': '300',
    # >   'num_days_in_episode': '1',
    # >   'start_timestamp': '2023-07-06 07:00:00+00:00'}

    # todo: generate_config(base_config_path, output_dir, params)
    modified_config = base_config  # consider passing the base_config instead

    filename_parts = []
    for i, param_name in enumerate(param_names):
      param_value = combination[i]
      modified_config = modify_config(modified_config, param_name, param_value)

      clean_name = param_filename_aliases.get(param_name) or param_name.replace('_', '')  # pylint:disable=line-too-long
      if param_name == 'start_timestamp':
        param_value = param_value.replace("'", '')
        filename_part = f'{clean_name}_{param_value[0:10]}'.replace('-', '')
      else:
        filename_part = f'{clean_name}_{param_value}'
      filename_parts.append(filename_part.strip())

    output_filename = f"{'_'.join(filename_parts)}.gin"
    # > "step_300_days_7_start_20230706.gin"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
      f.write(modified_config)

    logger.info('Generated: %s', output_path)


def main(argv: Sequence[str]):
  """When running absl app, we need the `argv` param, even though it is unused.

  See:

    + https://abseil.io/docs/python/guides/app
    + https://google.github.io/styleguide/pyguide.html#317-main
    + go/python-readability-advice#unused_argv
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  base_config_filepath = FLAGS.base_config
  output_dir = FLAGS.output_dir
  time_steps = FLAGS.time_steps
  num_days = FLAGS.num_days
  start_timestamps = FLAGS.start_timestamps

  # Handle both absolute and relative paths:
  if not os.path.isabs(base_config_filepath):
    logging.info('RELATIVE BASE CONFIG: %s', base_config_filepath)
    base_config_filepath = os.path.join(ROOT_DIR, base_config_filepath)

  if not os.path.isabs(output_dir):
    logging.info('RELATIVE OUTPUT DIR: %s', output_dir)
    output_dir = os.path.join(ROOT_DIR, output_dir)

  base_config_filepath = os.path.abspath(base_config_filepath)
  output_dir = os.path.abspath(output_dir)

  logging.info('Base Config Filepath: %s', base_config_filepath)
  logging.info('Output Dir: %s', output_dir)

  # Convert dates to datetimes:
  # start_timestamps = [f'{t.strip()} 07:00:00+00:00' for t in start_timestamps]
  # todo: get this to work without hard-coding in the extra quotes
  start_timestamps = [f"'{t.strip()} 07:00:00+00:00'" for t in start_timestamps]

  logging.info('Time Steps: %s', time_steps)
  logging.info('Num Days: %s', num_days)
  logging.info('Start Timestamps: %s', start_timestamps)

  params_grid = {
      'time_step_sec': time_steps,
      'num_days_in_episode': num_days,
      'start_timestamp': start_timestamps,
  }

  generate_configs(
      base_config_path=base_config_filepath,
      output_dir=output_dir,
      params_grid=params_grid,
  )

  logger.info(
      'Generated %d configuration files in %s',
      len(time_steps) * len(num_days),
      output_dir,
  )


if __name__ == '__main__':

  app.run(main)
