"""Utilities for generating example prompts.

Creates an example prompt and writes it to a markdown file in the "examples"
directory. This helps facilitate developer reviews of the prompt. Once written,
you can use the text editor's markdown preview functionality to view the prompt
and verify the formatting renders correctly.
"""

import os
from typing import Type

from absl import logging

from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.prompts import promptmaker


def write_prompt_md(
    promptmaker_class: Type[promptmaker.Promptmaker],
    include_weights: bool,
    dirpath: str,
    filename: str,
) -> None:
  """Generates an example prompt and writes it to a markdown file.

  Args:
    promptmaker_class: The promptmaker class to use.
    include_weights: Whether to include weights in the prompt.
    dirpath: The directory to write the markdown file to.
    filename: The name of the markdown file to write.
  """

  logging.info("LOADING ENVIRONMENT...")
  env = hybrid_action_environment.HybridActionEnvironment()
  logging.info("Current local timestamp: %s", env.current_local_timestamp)
  env.reset()

  logging.info("CREATING PROMPTMAKER: %s...", promptmaker_class.__name__)
  pm = promptmaker_class(
      env=env, include_weights=include_weights
  )

  logging.info("SETTING UP EXAMPLE PROMPTS DIRECTORY...")
  examples_dirpath = os.path.join(dirpath, "examples")
  os.makedirs(examples_dirpath, exist_ok=True)

  logging.info("WRITING PROMPT TO %s...", filename)
  md_filepath = os.path.join(examples_dirpath, filename)
  with open(md_filepath, "w") as f:
    f.write(pm.prompt)
    f.write("\n")

  logging.info("DONE")
