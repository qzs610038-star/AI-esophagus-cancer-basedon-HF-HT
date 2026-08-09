from scripts.pfmval_assets import build_shadow_inventory


def test_shadow_inventory_uses_lineage_and_does_not_hash_large_or_sensitive_files(
    tmp_path,
):
    root = tmp_path / "repo"
    (root / "pfmval_core").mkdir(parents=True)
    (root / "pfmval_core" / "metrics.py").write_text(
        "def compute_metrics(a, b): return {}\n",
        encoding="utf-8",
    )
    (root / "train_mpp_uni2h_mlp.py").write_text(
        "from pfmval_core.metrics import compute_metrics\n",
        encoding="utf-8",
    )
    (root / "ignored_checkpoints").mkdir()
    (root / "ignored_checkpoints" / "model.ckpt").write_bytes(b"x" * 1024)
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    inventory = build_shadow_inventory(root)
    by_path = {item["path"]: item for item in inventory["candidates"]}

    assert by_path["pfmval_core/metrics.py"]["era"] == "shared_dependency"
    assert by_path["pfmval_core/metrics.py"]["lifecycle"] == "active"
    assert "sha256" not in by_path["ignored_checkpoints/model.ckpt"]
    assert all(".env" not in item["path"] for item in inventory["candidates"])
    assert "secret" not in str(inventory)
