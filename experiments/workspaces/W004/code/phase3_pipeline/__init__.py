"""Local-only Phase 3 experiment components.

This package contains implementation code and synthetic smoke-test utilities.
Importing it never starts training, touches a server, or mutates project state.
"""

__all__ = ["PatchBag", "validate_fold_assignments", "validate_patch_bag"]


def __getattr__(name: str):
    """Load numerical contracts lazily so CSV-only server preflight is lightweight."""

    if name in __all__:
        from .contracts import PatchBag, validate_fold_assignments, validate_patch_bag

        return {
            "PatchBag": PatchBag,
            "validate_fold_assignments": validate_fold_assignments,
            "validate_patch_bag": validate_patch_bag,
        }[name]
    raise AttributeError(name)
