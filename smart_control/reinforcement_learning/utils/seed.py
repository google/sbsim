"""Utility for setting random seeds for reproducibility."""

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use (default: 42)
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
