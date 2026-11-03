"""Configuration-first alignment pipeline toolkit."""

# Import built-ins once so registry lookups work for both the Python API and
# the CLI.  External plugins can register additional entries after import.
from . import backends as _backends  # noqa: F401
from . import datasets as _datasets  # noqa: F401
from . import methods as _methods  # noqa: F401

from .config import load_experiment, load_yaml, resolve_experiment
from .datasets import AlignmentExample, DatasetDefinition, register_dataset
from .methods import MethodDefinition, register_method
from .pipeline import AlignmentExperiment, PipelineSpec
from .registry import BACKENDS, DATASETS, METHODS
from .trl_runner import TRLRunner

__all__ = [
    "AlignmentExample",
    "AlignmentExperiment",
    "BACKENDS",
    "DATASETS",
    "DatasetDefinition",
    "METHODS",
    "MethodDefinition",
    "PipelineSpec",
    "TRLRunner",
    "load_experiment",
    "load_yaml",
    "register_dataset",
    "register_method",
    "resolve_experiment",
]

__version__ = "0.1.0"
