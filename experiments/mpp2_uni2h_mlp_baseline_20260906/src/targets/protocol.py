from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


class MissingTargetContract(RuntimeError):
    """Raised when a reserved target is selected but required inputs are absent."""


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    names: Sequence[str]
    n_outputs: int
    label_kind: str
    source: str
    missing: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "names": list(self.names),
            "n_outputs": self.n_outputs,
            "label_kind": self.label_kind,
            "source": self.source,
            "missing": list(self.missing),
        }


class TargetAdapter:
    target_id = ""

    def spec(self) -> TargetSpec:
        raise NotImplementedError

    def label_csv(self, labels_root: Path, split: str, patient: str, train_mpp_id: int) -> Path:
        raise NotImplementedError
