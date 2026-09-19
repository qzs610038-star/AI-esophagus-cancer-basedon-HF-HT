from __future__ import annotations

from pathlib import Path


def test_default_launcher_delegates_only_to_seed42():
    package = Path(__file__).resolve().parents[1]
    content = (package / "run.ps1").read_text(encoding="utf-8")
    assert "run_seed42.ps1" in content
    assert "run_remaining_seeds.ps1" not in content


def test_remaining_seeds_launcher_requires_explicit_review_acknowledgement():
    package = Path(__file__).resolve().parents[1]
    content = (package / "run_remaining_seeds.ps1").read_text(encoding="utf-8")
    assert "ConfirmAfterSeed42Review" in content
    assert "--scope remaining" in content


def test_no_launcher_contains_resume_or_scientific_hyperparameter_switches():
    package = Path(__file__).resolve().parents[1]
    forbidden = ("--resume", "--learning-rate", "--dropout", "--hidden-dim")
    for path in package.glob("*.ps1"):
        text = path.read_text(encoding="utf-8").lower()
        assert all(value not in text for value in forbidden), path.name


def test_local_analysis_launcher_accepts_seed42_and_remaining_batches_together():
    package = Path(__file__).resolve().parents[1]
    content = (package / "analyze_local.ps1").read_text(encoding="utf-8")
    assert "[string[]]$BatchDir" in content
    assert "foreach ($value in $BatchDir)" in content
