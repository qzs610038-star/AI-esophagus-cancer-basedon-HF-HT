"""Package-local fail-closed errors. Copied contract from the full-FOV package plus download/baseline errors."""

from __future__ import annotations


class ConfigError(ValueError):
    """Unsupported algorithm, missing required field, or invalid range."""


class IdentityMismatchError(RuntimeError):
    """Feature/label/manifest identity keys do not match."""

    error_code = "IDENTITY_MISMATCH"


class NonFiniteDataError(RuntimeError):
    """NaN or Inf in features or labels."""

    error_code = "NONFINITE_DATA"


class SlideMappingMissingError(RuntimeError):
    """Verified (patient_id, slide_id) mapping is required but absent."""

    error_code = "SLIDE_MAPPING_UNVERIFIED"


class UnsupportedAlgorithmError(ConfigError):
    """A config switch names an algorithm that this package does not implement."""


class OfflineModelError(ConfigError):
    """A model cannot be constructed strictly from the registered local snapshot."""


class BaselineReferenceError(RuntimeError):
    """Accepted full-FOV spatial baselines are missing, duplicated, or not spatial."""

    error_code = "BASELINE_REFERENCE"


class AccessDeniedError(OfflineModelError):
    """Hugging Face 401/403 or gated-repo denial; not a retryable network blip."""
