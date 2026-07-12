"""AI Incidents pipeline package.

Provides shared setup helpers (logging configuration and global seeding)
used by the pipeline runner before invoking the individual stages.
"""

import logging
import random

import numpy as np


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for the pipeline.

    Args:
        level: Logging level to apply to the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def set_global_seeds(seed: int) -> None:
    """Fix random seeds across libraries for reproducibility.

    Args:
        seed: Seed value applied to ``random``, ``numpy`` and (if installed)
            ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
