"""Experiment-specific fail-closed exceptions."""


class ExperimentError(RuntimeError):
    """Base class for errors that invalidate an experiment unit."""


class ConfigError(ExperimentError):
    """The frozen experiment contract or runtime configuration is invalid."""


class IdentityMismatchError(ExperimentError):
    """Images, features, labels, or predictions do not share the same identities."""


class NonFiniteDataError(ExperimentError):
    """An input or output contains NaN or infinite values."""


class ReferenceMismatchError(ExperimentError):
    """A historical UNI2-h result is not a matching point reference."""


class OfflineModelError(ExperimentError):
    """A model cannot be constructed strictly from the registered local snapshot."""
