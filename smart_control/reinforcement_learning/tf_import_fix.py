"""Fixes known issue when importing tensorflow.

Import this before importing tensorflow:

    import smart_control.reinforcement_learning.tf_import_fix
"""

# ISSUE:
# OK so we are running into an error when using tf_agents:
# TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
# https://github.com/tensorflow/tensorflow/issues/59869

# SOLUTION:
# As a workaround, we need to set this env var before loading tensorflow
# https://github.com/GrahamDumpleton/wrapt/issues/231#issuecomment-1455800902

import os

os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
