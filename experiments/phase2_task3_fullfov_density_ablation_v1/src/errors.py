"""Shared errors for the Phase2 softlink v2.1 core package."""

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
