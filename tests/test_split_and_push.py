import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

EXPECTED_COLUMNS = ["text", "rating", "title", "asin", "product_category"]


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "split_and_push_to_hf.py"
    spec = importlib.util.spec_from_file_location("split_and_push_to_hf", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "failed to load split_and_push_to_hf module"
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_split_and_push_generates_valid_parquet(tmp_path):
    module = _load_script_module()

    shard_path = tmp_path / "data" / "snapshot_0000.jsonl"
    rows = [
        {
            "text": "Great battery life",
            "rating": 5,
            "title": "Battery",
            "asin": "ASIN001",
            "product_category": "Electronics",
        },
        {
            "text": "Terrible quality",
            "rating": 1,
            "title": "Quality",
            "asin": "ASIN002",
            "product_category": "Electronics",
        },
    ]
    _write_jsonl(shard_path, rows)

    manifest = {"files": [str(shard_path.relative_to(tmp_path))]}
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "splits"
    cfg = {
        "manifest_path": str(manifest_path),
        "base_dir": str(tmp_path),
        "categories": ["Electronics"],
        "dev": {"per_category": 1},
        "eval": {"per_category": 1},
        "seed": 123,
        "dedup_by_id": True,
        "out": {
            "dev_jsonl": str(out_dir / "dev.jsonl"),
            "eval_jsonl": str(out_dir / "eval.jsonl"),
            "dev_parquet": str(out_dir / "dev.parquet"),
            "eval_parquet": str(out_dir / "eval.parquet"),
            "blocklist_out": str(tmp_path / "blocklist.txt"),
        },
        "hf": {"push": False},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    diagnostics = module.run_from_config(str(cfg_path))
    assert diagnostics["dev_rows"] == 1
    assert diagnostics["eval_rows"] == 1

    dev_df = pd.read_parquet(out_dir / "dev.parquet")
    eval_df = pd.read_parquet(out_dir / "eval.parquet")

    for frame in (dev_df, eval_df):
        assert list(frame.columns) == EXPECTED_COLUMNS
        assert not frame.isna().any().any()
        assert frame["rating"].between(0, 5).all()
        assert frame["text"].str.len().min() > 0
        assert frame["asin"].str.startswith("ASIN").all()

    blocklist_content = (
        tmp_path / "blocklist.txt"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(blocklist_content) == diagnostics["dev_rows"] + diagnostics["eval_rows"]
