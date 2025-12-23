"""PETase model package.

Primary API endpoints for training and using PETase surrogate ensembles.
"""

from .surrogate import SurrogateEnsemble, SurrogateConfig, load_config, train_from_config

__all__ = [
    "SurrogateEnsemble",
    "SurrogateConfig",
    "load_config",
    "train_from_config",
]
