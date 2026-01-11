"""Create stratified dev/eval splits from JSONL shards and optionally push to Hugging Face."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

try:  # Optional heavy dependencies; imported lazily for tests
    import pandas as pd
    from datasets import Dataset, DatasetDict
except Exception as exc:  # pragma: no cover - import guard for optional deps
    print(
        "Please install dependencies: pip install datasets pandas pyarrow fastparquet",
        f"(error: {exc})",
        file=sys.stderr,
    )
    raise

_WS = re.compile(r"\s+", flags=re.U)


def stable_id(text: str, asin: str) -> str:
    """Stable SHA1 hash identical to ``restar.utils.stable_id``."""

    t = _WS.sub(" ", (text or "").strip()).lower()
    a = (asin or "").strip().lower()
    raw = f"{t}\x01{a}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_block_ids(paths: List[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths or []:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    ids.add(value)
    return ids


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except Exception:
                continue


def reservoir_update(
    reservoir: List[Any],
    target_k: int,
    new_item: Any,
    seen_n: int,
    rng: random.Random,
) -> None:
    """Standard reservoir sampling with ``seen_n`` starting at 1."""

    if len(reservoir) < target_k:
        reservoir.append(new_item)
        return

    idx = rng.randint(0, seen_n - 1)
    if idx < target_k:
        reservoir[idx] = new_item


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_splits(
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(cfg.get("seed", 42)))

    manifest_path = cfg["manifest_path"]
    base_dir = cfg.get("base_dir") or os.path.dirname(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    files = [
        path if os.path.isabs(path) else os.path.join(base_dir, path)
        for path in manifest.get("files", [])
    ]

    categories_cfg = cfg.get("categories") or []
    dev_k = int(cfg.get("dev", {}).get("per_category", 0))
    eval_k = int(cfg.get("eval", {}).get("per_category", 0))
    target_k = dev_k + eval_k

    pre_block = load_block_ids(cfg.get("id_blocklist_in", []))
    dedup = bool(cfg.get("dedup_by_id", True))

    reservoirs: Dict[str, List[Dict[str, Any]]] = {}
    seen_counts: Dict[str, int] = {}
    seen_ids: set[str] = set()

    def accept(row: Dict[str, Any]) -> Tuple[bool, str, str]:
        category = row.get("product_category", "") or ""
        if categories_cfg and category not in categories_cfg:
            return False, "", ""
        rid = stable_id(row.get("text", ""), row.get("asin", ""))
        if rid in pre_block:
            return False, "", ""
        if dedup and rid in seen_ids:
            return False, "", ""
        return True, category, rid

    for path in files:
        if not os.path.exists(path):
            print(f"[warn] missing shard: {path}", file=sys.stderr)
            continue
        for row in iter_jsonl(path):
            ok, category, rid = accept(row)
            if not ok:
                continue

            buffer = reservoirs.setdefault(category, [])
            seen_counts[category] = seen_counts.get(category, 0) + 1

            normalized = {
                "text": row.get("text", "") or "",
                "rating": row.get("rating", 0) or 0,
                "title": row.get("title", "") or "",
                "asin": row.get("asin", "") or "",
                "product_category": category,
            }
            reservoir_update(buffer, target_k, normalized, seen_counts[category], rng)
            if dedup:
                seen_ids.add(rid)

    dev_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for category, buffer in reservoirs.items():
        rng.shuffle(buffer)
        dev_take = min(dev_k, len(buffer))
        eval_take = min(eval_k, max(0, len(buffer) - dev_take))
        dev_rows.extend(buffer[:dev_take])
        eval_rows.extend(buffer[dev_take : dev_take + eval_take])
        if dev_take < dev_k or eval_take < eval_k:
            print(
                f"[warn] category={category} short: dev={dev_take}/{dev_k}, eval={eval_take}/{eval_k}",
                file=sys.stderr,
            )

    diagnostics = {
        "dev_rows": len(dev_rows),
        "eval_rows": len(eval_rows),
        "categories": {category: len(buffer) for category, buffer in reservoirs.items()},
    }
    return dev_rows, eval_rows, diagnostics


def dump_outputs(
    cfg: Dict[str, Any],
    dev_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
) -> None:
    out_cfg = cfg["out"]

    def ensure_parent(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
        ensure_parent(path)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump_jsonl(out_cfg["dev_jsonl"], dev_rows)
    dump_jsonl(out_cfg["eval_jsonl"], eval_rows)

    def maybe_dump_parquet(key: str, rows: List[Dict[str, Any]]) -> None:
        target = out_cfg.get(key)
        if not target:
            return
        ensure_parent(target)
        frame = pd.DataFrame(rows, columns=["text", "rating", "title", "asin", "product_category"])
        frame.to_parquet(target, index=False)

    maybe_dump_parquet("dev_parquet", dev_rows)
    maybe_dump_parquet("eval_parquet", eval_rows)

    block_path = out_cfg.get("blocklist_out")
    if block_path:
        ensure_parent(block_path)
        with open(block_path, "w", encoding="utf-8") as handle:
            for row in dev_rows + eval_rows:
                handle.write(stable_id(row["text"], row["asin"]) + "\n")

    manifest_eval = out_cfg.get("manifest_eval")
    if manifest_eval:
        ensure_parent(manifest_eval)
        with open(manifest_eval, "w", encoding="utf-8") as handle:
            json.dump({"files": [out_cfg["eval_jsonl"]]}, handle, ensure_ascii=False, indent=2)

    hf_cfg = cfg.get("hf", {}) or {}
    if hf_cfg.get("push", False):
        repo_id = hf_cfg["repo_id"]
        private = bool(hf_cfg.get("private", True))
        token = None
        token_env = hf_cfg.get("token_env")
        if token_env:
            token = os.environ.get(token_env)

        dataset = DatasetDict(
            {
                "dev": Dataset.from_pandas(pd.DataFrame(dev_rows), preserve_index=False),
                "eval": Dataset.from_pandas(pd.DataFrame(eval_rows), preserve_index=False),
            }
        )
        dataset.push_to_hub(
            repo_id,
            token=token,
            private=private,
            commit_message=hf_cfg.get("commit_message", "Add dev/eval splits"),
        )


def run_from_config(path: str) -> Dict[str, Any]:
    cfg = _load_config(path)
    dev_rows, eval_rows, diagnostics = build_splits(cfg)
    dump_outputs(cfg, dev_rows, eval_rows)
    diagnostics.update(
        {
            "dev_output": cfg["out"].get("dev_jsonl"),
            "eval_output": cfg["out"].get("eval_jsonl"),
        }
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    summary = run_from_config(args.config)
    print("[done]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
