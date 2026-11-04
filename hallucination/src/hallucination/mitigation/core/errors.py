from __future__ import annotations


class HallucinationMitigationError(Exception):
    """Base exception for the toolkit."""


class ConfigError(HallucinationMitigationError):
    """Invalid method, preset, or user configuration."""


class MissingDependencyError(HallucinationMitigationError):
    """An optional dependency is required for the requested feature."""


class UnsupportedModelError(HallucinationMitigationError):
    """The model or backend cannot support the requested method."""


class ArtifactNotFoundError(HallucinationMitigationError):
    """A steering/probe artifact is required but was not provided."""


class ArtifactMismatchError(HallucinationMitigationError):
    """An artifact does not match the loaded model."""


class BackendError(HallucinationMitigationError):
    """The model backend (API or local runtime) failed."""


class GenerationError(HallucinationMitigationError):
    """Generation failed or produced an unusable output."""
