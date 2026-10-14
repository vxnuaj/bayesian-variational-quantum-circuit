"""Bayesian-optimized variational quantum circuits for toy classification."""
from .data import make_dataset
from .circuit import VQC
from .optimize import bayesian_optimize, block_bayesian_optimize, random_search, local_polish, cobyla_from_random
from .classical import classical_baselines

__all__ = [
    "make_dataset", "VQC", "bayesian_optimize", "block_bayesian_optimize", "random_search",
    "local_polish", "cobyla_from_random", "classical_baselines",
]
