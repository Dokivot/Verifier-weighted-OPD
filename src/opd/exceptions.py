class OPDError(RuntimeError):
    """Base exception for expected OPD pipeline failures."""


class ConfigurationError(OPDError):
    """Raised when a configuration is missing or invalid."""


class DependencyError(OPDError):
    """Raised when an optional production dependency is unavailable."""


class ArtifactError(OPDError):
    """Raised when an artifact is invalid or incomplete."""
