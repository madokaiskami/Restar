"""Interactive labeling tool that writes incremental Parquet shards."""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set

# -*- coding: utf-8 -*-
"""Interactive labeling tool that writes incremental Parquet shards.

* Input: JSONL produced by ``restar.abstain_stream`` (``to_label.jsonl`` or ``*.partial.jsonl``)
* Output: ``output/human_labeled/human_labeled-*.parquet``
* Resumable: previously labeled IDs are skipped automatically

Dependencies: ``pip install pyarrow datasets``
"""
import argparse, json, os, sys, time, glob
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

WS = re.compile(r"\s+", flags=re.U)


def map_rating_to_label(rating: Any) -> Optional[int]:
    """Convert a raw rating value into a coarse label."""

    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if value <= 2:
        return 0
    if value >= 4:
        return 2
    return 1


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(WS.sub(" ", text).strip().split())


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def load_existing_ids(parquet_dir: str) -> Set[str]:
    ids: Set[str] = set()
    if not os.path.isdir(parquet_dir):
        return ids
    try:
        dataset = ds.dataset(parquet_dir, format="parquet")
    # Read only the ``id`` column to keep memory usage low
    try:
        dataset = ds.dataset(parquet_dir, format="parquet")
        # Iterate batch by batch instead of loading everything at once
        scanner = dataset.scan(columns=["id"])
        for batch in scanner.to_batches():
            ids.update(v for v in batch.column(0).to_pylist() if v)
    except Exception as exc:  # pragma: no cover - observational logging
        print(f"[warn] failed to scan existing parquet ids: {exc}")
    return ids


def make_schema() -> pa.schema:
    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("title", pa.string()),
            ("asin", pa.string()),
            ("product_category", pa.string()),
            ("rating", pa.float32()),
            ("pred_label", pa.int64()),
            ("probs", pa.string()),
            ("uncertainty", pa.float32()),
            ("label", pa.int64()),
        ]
    )


def normalize_probs(probs: Any) -> str:
    if probs is None:
        return "[]"
    if isinstance(probs, str):
        try:
            json.loads(probs)
            return probs
        except json.JSONDecodeError:
            parts = [p.strip() for p in probs.split(",") if p.strip()]
    return pa.schema([
        ("id", pa.string()),
        ("text", pa.string()),
        ("title", pa.string()),
        ("asin", pa.string()),
        ("product_category", pa.string()),
        ("rating", pa.float32()),
        ("pred_label", pa.int64()),
        ("probs", pa.string()),         # Store JSON strings for compatibility
        ("uncertainty", pa.float32()),
        ("label", pa.int64()),          # Human label
    ])

def records_to_table(recs: List[Dict[str, Any]]) -> pa.Table:
    # Normalize the ``probs`` field to a JSON string
    def norm_probs(p):
        if p is None:
            return "[]"
        if isinstance(p, str):
            # Try validating JSON; otherwise convert best-effort
            try:
                _ = json.loads(p)
                return p
            except:
                try:
                    return json.dumps([float(x) for x in p.split(",")])
                except:
                    return "[]"
        if isinstance(p, (list, tuple)):
            try:
                return json.dumps([float(x) for x in parts])
            except ValueError:
                return "[]"
    if isinstance(probs, (list, tuple)):
        try:
            return json.dumps([float(x) for x in probs])
        except (TypeError, ValueError):
            return "[]"
    return "[]"


def records_to_table(records: List[Dict[str, Any]]) -> pa.Table:
    columns: Dict[str, List[Any]] = {key: [] for key in make_schema().names}
    for record in records:
        columns["id"].append(record.get("id", ""))
        columns["text"].append(record.get("text", ""))
        columns["title"].append(record.get("title", ""))
        columns["asin"].append(record.get("asin", ""))
        columns["product_category"].append(record.get("product_category", ""))

        try:
            columns["rating"].append(float(record.get("rating", 0.0)))
        except (TypeError, ValueError):
            columns["rating"].append(0.0)


        try:
            columns["rating"].append(float(record.get("rating", 0.0)))
        except (TypeError, ValueError):
            columns["rating"].append(0.0)

    cols = {
        "id": [],
        "text": [],
        "title": [],
        "asin": [],
        "product_category": [],
        "rating": [],
        "pred_label": [],
        "probs": [],
        "uncertainty": [],
        "label": [],
    }
    for r in recs:
        cols["id"].append(r.get("id", ""))
        cols["text"].append(r.get("text", ""))
        cols["title"].append(r.get("title", ""))
        cols["asin"].append(r.get("asin", ""))
        cols["product_category"].append(r.get("product_category", ""))
        # ``rating`` may be a string/None
        try:
            cols["rating"].append(float(r.get("rating", 0.0)))
        except:
            cols["rating"].append(float(0.0))
        # Prediction metadata
        try:
            columns["pred_label"].append(int(record.get("pred_label", -1)))
        except (TypeError, ValueError):
            columns["pred_label"].append(-1)

        columns["probs"].append(normalize_probs(record.get("probs")))

        try:
            columns["uncertainty"].append(float(record.get("uncertainty", 0.0)))
        except (TypeError, ValueError):
            columns["uncertainty"].append(0.0)

        columns["label"].append(int(record["label"]))
    return pa.table(columns, schema=make_schema())


def write_chunk_to_dataset(records: List[Dict[str, Any]], out_dir: str, chunk_idx: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    table = records_to_table(records)
    os.makedirs(out_dir, exist_ok=True)
    table = records_to_table(records)
            cols["uncertainty"].append(float(r.get("uncertainty", 0.0)))
        except:
            cols["uncertainty"].append(float(0.0))
        # Human-provided label
        cols["label"].append(int(r["label"]))
    table = pa.table(cols, schema=make_schema())
    return table

def write_chunk_to_dataset(recs: List[Dict[str, Any]], out_dir: str, chunk_idx: int):
    os.makedirs(out_dir, exist_ok=True)
    table = records_to_table(recs)
    # Rolling file names: human_labeled-00001.parquet
    path = os.path.join(out_dir, f"human_labeled-{chunk_idx:05d}.parquet")
    pq.write_table(table, path, compression="zstd")
    return path


def next_chunk_index(out_dir: str) -> int:
    files = sorted(glob.glob(os.path.join(out_dir, "human_labeled-*.parquet")))
    if not files:
        return 1
    suffix = files[-1].rsplit("-", 1)[-1].replace(".parquet", "")
    # Use the highest index + 1 for the next shard
    last = files[-1].rsplit("-", 1)[-1].replace(".parquet", "")
    try:
        return int(suffix) + 1
    except ValueError:
        return len(files) + 1


def prompt_label(default_label: Optional[int]) -> Optional[int]:
    while True:
        try:
            value = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise
        if value == "q":
            return -1
        if value == "s":
            return None
        if value == "":
            return default_label
        if value in {"0", "1", "2"}:
            return int(value)
        print("Invalid input. Enter 0/1/2/s/q or press Enter for default.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="JSONL from restar.abstain_stream")
    parser.add_argument(
        "--out_dir",
        default="output/human_labeled",
        help="Directory for Parquet shards",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Number of rows per Parquet shard",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=400,
        help="Truncate display text to this length",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=0,
        help="Skip samples shorter than this word count",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    os.makedirs(args.out_dir, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="JSONL from restar.abstain_stream")
    ap.add_argument("--out_dir", default="output/human_labeled", help="Directory for Parquet shards")
    ap.add_argument("--chunk_size", type=int, default=200, help="Number of rows per Parquet shard")
    ap.add_argument("--max_len", type=int, default=400, help="Truncate display text to this length")
    ap.add_argument("--min_words", type=int, default=0, help="Skip samples shorter than this word count")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resume mode: skip IDs that already exist
    existing_ids = load_existing_ids(args.out_dir)
    print(f"[info] resume with existing labeled ids: {len(existing_ids)}")

    buffer: List[Dict[str, Any]] = []
    chunk_idx = next_chunk_index(args.out_dir)
    processed = 0
    written = 0

    for record in iter_jsonl(args.in_jsonl):
        processed += 1
        sample_id = record.get("id")
        if not sample_id or sample_id in existing_ids:
            continue

        text = (
            (record.get("title") or "").strip() + "  " + (record.get("text") or "").strip()
        ).strip()
        if args.min_words and word_count(text) < args.min_words:
            continue

        text = (
            (record.get("title") or "").strip()
            + "  "
            + (record.get("text") or "").strip()
        ).strip()
        if args.min_words and word_count(text) < args.min_words:
        sid = ex.get("id", None)
        if not sid:
            # Skip entries without IDs
            continue
        if sid in existing_ids:
            # Skip IDs that were already labeled
            continue

        # Compose display text
        text = ((ex.get("title") or "").strip() + "  " + (ex.get("text") or "").strip()).strip()
        if args.min_words and _wc(text) < args.min_words:
            continue

        display_text = f"{text[:args.max_len]} ..." if len(text) > args.max_len else text
        rating = record.get("rating")
        pred_label = record.get("pred_label")
        default_label = map_rating_to_label(rating)

        print("=" * 100)
        print(
            f"[{processed}] id={sample_id} asin={record.get('asin', '')} "
            f"cat={record.get('product_category', '')}"
        )
        if rating is not None:
            print(f"rating={rating}  pred_label={pred_label}  default_by_rating={default_label}")
        print(display_text)
        print(
            "Enter label: 0=neg, 1=neutral, 2=pos, s=skip, q=quit; "
            "Enter to use default_by_rating"
        )

        try:
            label = prompt_label(default_label)
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] Interrupted. Flushing buffer...")
            if buffer:
                path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
                print(f"[flush] wrote {len(buffer)} rows -> {path}")
            print(f"[done] processed={processed}, written={written}")
            sys.exit(0)

        if label == -1:
            print("[exit] User requested exit. Flushing buffer...")
            if buffer:
                path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
                print(f"[flush] wrote {len(buffer)} rows -> {path}")
            print(f"[done] processed={processed}, written={written}")
            sys.exit(0)

        if label is None:
            continue

        record = dict(record)
        record["label"] = int(label)
        buffer.append(record)
        existing_ids.add(sample_id)

        try:
            label = prompt_label(default_label)
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] Interrupted. Flushing buffer...")
            if buffer:
                path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
                print(f"[flush] wrote {len(buffer)} rows -> {path}")
            print(f"[done] processed={processed}, written={written}")
            sys.exit(0)

        if label == -1:
            print("[exit] User requested exit. Flushing buffer...")
            if buffer:
                path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
                print(f"[flush] wrote {len(buffer)} rows -> {path}")
            print(f"[done] processed={processed}, written={written}")
            sys.exit(0)

        if label is None:
            continue

        record = dict(record)
        record["label"] = int(label)
        buffer.append(record)
        existing_ids.add(sample_id)
        print(text_disp)
        print("Enter label: 0=neg, 1=neutral, 2=pos, s=skip, q=quit; Enter to use default_by_rating")

        while True:
            try:
                inp = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[exit] Interrupted. Flushing buffer...")
                if buf:
                    path = write_chunk_to_dataset(buf, args.out_dir, chunk_idx)
                    print(f"[flush] wrote {len(buf)} rows -> {path}")
                print(f"[done] processed={processed}, written={written}")
                sys.exit(0)

            if inp == "q":
                print("[exit] User requested exit. Flushing buffer...")
                if buf:
                    path = write_chunk_to_dataset(buf, args.out_dir, chunk_idx)
                    print(f"[flush] wrote {len(buf)} rows -> {path}")
                print(f"[done] processed={processed}, written={written}")
                sys.exit(0)
            if inp == "s":
                lab = None
                break
            if inp == "":
                lab = default_label
                break
            if inp in {"0", "1", "2"}:
                lab = int(inp)
                break
            print("Invalid input. Enter 0/1/2/s/q or press Enter.")

        if lab is None:
            continue

        rec = dict(ex)
        rec["label"] = int(lab)
        buf.append(rec)
        existing_ids.add(sid)  # Update de-duplication cache immediately

        if len(buffer) >= args.chunk_size:
            path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
            written += len(buffer)
            print(f"[write] chunk#{chunk_idx} rows={len(buffer)} -> {path}")
            buffer.clear()
            chunk_idx += 1

    if buffer:
        path = write_chunk_to_dataset(buffer, args.out_dir, chunk_idx)
        written += len(buffer)
        print(f"[write] chunk#{chunk_idx} rows={len(buffer)} -> {path}")
    # Flush remaining records
    if buf:
        path = write_chunk_to_dataset(buf, args.out_dir, chunk_idx)
        written += len(buf)
        print(f"[write] chunk#{chunk_idx} rows={len(buf)} -> {path}")

    print(f"[done] processed={processed}, written={written}, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
