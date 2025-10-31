"""PETase model package.

This module re-exports the primary API endpoints for training and using
PETase surrogate models. Direct usage:
    from petase.models import SurrogateModel, load_config
"""

from .surrogate import SurrogateModel, SurrogateConfig, load_config

__all__ = [
    "SurrogateModel",
    "SurrogateConfig",
    "load_config",
]
