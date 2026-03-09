"""Reproducibility utilities: fixed seeds for numpy and random."""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy and Python random to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a numpy Generator with the given seed (or default)."""
    return np.random.default_rng(seed)
