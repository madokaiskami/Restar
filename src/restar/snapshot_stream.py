# src/restar/snapshot_stream.py
import argparse
import gzip
import json
import os

from datasets import load_dataset


def _stream_category(cat: str, n: int, seed: int, bufsize: int = 10000):
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{cat}",
        streaming=True,
        trust_remote_code=True,
    )["full"].shuffle(seed=seed, buffer_size=bufsize)
    c = 0
    for ex in ds:
        # Normalize the required fields and provide fallbacks
        yield {
            "text": ex.get("text", "") or "",
            "rating": int(ex.get("rating", 0)),
            "title": ex.get("title", "") or "",
            "asin": ex.get("asin", "") or "",
            "product_category": cat,
        }
        c += 1
        if c >= n:
            break


def _rotate(out_dir: str, shard_idx: int):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"snapshot_{shard_idx:04d}.jsonl.gz")
    return gzip.open(path, "wt", encoding="utf-8"), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard_size", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from .config import load_config

    cfg = load_config(args.config)
    out_dir = os.path.join("data", "snapshots", cfg.run_name)
    cats = cfg.data.categories
    n_per = int(cfg.data.len_each_category)

    shard_size = int(args.shard_size)
    writer, cur_path = _rotate(out_dir, shard_idx=0)
    shard_idx, in_shard = 0, 0
    files = []

    try:
        for cat in cats:
            for row in _stream_category(cat, n_per, seed=args.seed):
                # Write line by line to avoid loading everything into memory
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                in_shard += 1
                if in_shard >= shard_size:
                    writer.close()
                    files.append(cur_path)
                    shard_idx += 1
                    writer, cur_path = _rotate(out_dir, shard_idx)
                    in_shard = 0
        # Close the final shard
        writer.close()
        files.append(cur_path)
    finally:
        try:
            writer.close()
        except Exception:
            pass

    manifest = {
        "run_name": cfg.run_name,
        "categories": cats,
        "len_each_category": n_per,
        "total": n_per * len(cats),
        "files": [os.path.basename(p) for p in files],
        "schema": ["text", "rating", "title", "asin", "product_category"],
        "shard_size": shard_size,
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
