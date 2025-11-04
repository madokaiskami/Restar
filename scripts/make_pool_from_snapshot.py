"""Convert snapshot shards into a single unlabeled pool CSV."""

import argparse
import gzip
import json
import os
from typing import List

import pandas as pd


def load_rows(snapshot_dir: str, limit_per_file: int | None) -> List[dict]:
    rows: List[dict] = []
    for filename in sorted(os.listdir(snapshot_dir)):
        if not filename.endswith(".jsonl.gz"):
            continue
        path = os.path.join(snapshot_dir, filename)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if limit_per_file is not None and idx >= limit_per_file:
                    break
                payload = json.loads(line)
                rows.append(
                    {
                        "id": f"{filename}:{idx}",
                        "text": payload.get("text", ""),
                        "title": payload.get("title", ""),
                        "asin": payload.get("asin", ""),
                        "product_category": payload.get("product_category", ""),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", default="data/snapshots/default")
    parser.add_argument("--out", default="data/pool/pool_unlabeled.csv")
    parser.add_argument("--limit_per_file", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records = load_rows(args.snapshot_dir, args.limit_per_file)
    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"[ok] pool size={len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
