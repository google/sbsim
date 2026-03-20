"""Example prompt generator for Building 'SB-1'.

Creates an example prompt and writes it to a markdown file in the "examples"
directory. This helps facilitate developer reviews of the prompt. Once written,
you can use the text editor's markdown preview functionality to view the prompt
and verify the formatting renders correctly.

To run this script using blaze:

```sh
blaze run //third_party/py/smart_buildings/smart_control/llm/prompts/sb1:example_prompt_generator
```

Arguments:

  --include_weights: Whether to include weights in the prompt (default: True).
  --md_filename: Filename for the markdown file (default: 'example_prompt.md').
"""  # pylint: disable=line-too-long

import os

from absl import app
from absl import flags

from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.prompts.sb1 import sb1_promptmaker
from smart_buildings.smart_control.utils.config_utils import full_config

INCLUDE_WEIGHTS = flags.DEFINE_boolean(
    "include_weights", True, "Include weights in the prompt."
)
MD_FILENAME = flags.DEFINE_string(
    "md_filename", "example_prompt.md", "Filename for the markdown file.",
)


def main(_) -> None:
  """Loads environment, creates prompt, and writes to markdown file.

  Uses a fully configured environment, and a prompt that has weights included.
  """

  print("SETTING GIN CONFIG...")
  full_config.set_gin_config()

  print("LOADING ENVIRONMENT...")
  env = hybrid_action_environment.HybridActionEnvironment()
  print(env.current_local_timestamp)
  env.reset()

  print("CREATING PROMPTMAKER...")
  pm = sb1_promptmaker.SB1Promptmaker(
      env, include_weights=INCLUDE_WEIGHTS.value
  )

  print("SETTING UP EXAMPLE PROMPTS DIRECTORY...")
  dirpath = os.path.dirname(os.path.realpath(__file__))
  print(dirpath)
  examples_dirpath = os.path.join(dirpath, "examples")
  os.makedirs(examples_dirpath, exist_ok=True)

  print("WRITING PROMPT TO MARKDOWN FILE...")
  md_filepath = os.path.join(examples_dirpath, MD_FILENAME.value)
  with open(md_filepath, "w") as f:
    f.write(pm.prompt)
    f.write("\n")

  print("DONE")


if __name__ == "__main__":
  app.run(main)
