"""Example prompt generator for Building 'SB-1'.

To run this script using blaze:

```sh
blaze run //third_party/py/smart_buildings/smart_control/llm/prompts/sb1:generator
```

Arguments:

  --include_weights: Whether to include weights in the prompt (default: True).
  --md_filename: Filename for the markdown file (default: 'example_prompt.md').
"""  # pylint: disable=line-too-long

import os

from absl import app
from absl import flags

from smart_buildings.smart_control.configs.resources.sb1.config_utils import full_config
from smart_buildings.smart_control.llm.prompts import generator
from smart_buildings.smart_control.llm.prompts.sb1 import sb1_promptmaker

INCLUDE_WEIGHTS = flags.DEFINE_boolean(
    "include_weights", True, "Include weights in the prompt."
)


def main(_) -> None:
  """Loads environment, creates prompt, and writes to markdown file."""

  print("SETTING GIN CONFIG...")
  full_config.set_gin_config()

  generator.write_prompt_md(
      promptmaker_class=sb1_promptmaker.SB1Promptmaker,
      include_weights=INCLUDE_WEIGHTS.value,
      dirpath=os.path.dirname(os.path.realpath(__file__)),
      filename="example_prompt.md",
  )

  generator.write_prompt_md(
      promptmaker_class=sb1_promptmaker.SB1FloorBasedPromptmaker,
      include_weights=INCLUDE_WEIGHTS.value,
      dirpath=os.path.dirname(os.path.realpath(__file__)),
      filename="example_floor_based_prompt.md",
  )


if __name__ == "__main__":
  app.run(main)
