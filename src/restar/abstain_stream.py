# src/restar/abstain_stream.py
import argparse
import heapq
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset, Features, Value, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import load_config
from .utils import append_ids, maybe_load_blocklist, set_seed, stable_id

_WS = re.compile(r"\s+", flags=re.U)


def _wc(text: str) -> int:
    if not text:
        return 0
    return len(_WS.sub(" ", text).strip().split())


def _iter_hf_amazon_stream(categories, seed, buffer_size) -> Iterable[Dict[str, Any]]:
    for cat in categories:
        cfg = f"raw_review_{cat}"
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            cfg,
            streaming=True,
            trust_remote_code=True,  # Prevent interactive prompts
        )["full"].shuffle(seed=seed, buffer_size=buffer_size)
        for ex in ds:
            yield {
                "text": ex.get("text", "") or "",
                "rating": ex.get("rating", 0.0),
                "title": ex.get("title", "") or "",
                "asin": ex.get("asin", "") or "",
                "product_category": cat,
            }


def _label_from_rating(value) -> int:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        rating = 0.0
    if rating <= 2:
        return 0
    if rating >= 4:
        return 2
    return 1


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _margin(p: np.ndarray) -> float:
    s = np.sort(p)[::-1]
    if len(s) < 2:
        return 1.0
    return float(s[0] - s[1])


def _uncertainty(p: np.ndarray) -> float:
    # Larger values mean higher uncertainty
    H = _entropy(p)
    M = _margin(p)
    return H + (1.0 - M)


class TopK:
    """Min-heap that keeps the top-K items by score and reports replacements."""

    def __init__(self, k: int):
        self.k = k
        self.heap: List[Tuple[float, int, Dict[str, Any]]] = []
        self._cnt = 0

    def push(self, score: float, payload: Dict[str, Any]) -> bool:
        """Return True when the heap changes (insert or replace)."""
        item = (score, self._cnt, payload)
        self._cnt += 1
        if self.k <= 0:
            return False
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
            return True
        if score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)
            return True
        return False

    def result(self) -> List[Dict[str, Any]]:
        return [p for (_, _, p) in sorted(self.heap, key=lambda x: (-x[0], x[1]))]

    def full(self) -> bool:
        return len(self.heap) >= self.k


class PerClassQuota:
    """Selector that enforces per-class quotas."""

    def __init__(self, quotas: List[int]):
        if len(quotas) != 3:
            raise ValueError("per-class quotas must contain exactly three values")
        self.bins = [TopK(q) for q in quotas]

    def push(self, y_pred: int, score: float, payload: Dict[str, Any]) -> bool:
        return self.bins[y_pred].push(score, payload)

    def result(self) -> List[Dict[str, Any]]:
        out = []
        for b in self.bins:
            out.extend(b.result())
        return out

    def all_full(self) -> bool:
        return all(b.full() for b in self.bins)


def _save_outputs(picked: List[Dict[str, Any]], out_jsonl: str, out_parquet: str):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for x in picked:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    feats = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "title": Value("string"),
            "asin": Value("string"),
            "product_category": Value("string"),
            "rating": Value("float32"),
            "pred_label": Value("int64"),
            "probs": Value("string"),
            "uncertainty": Value("float32"),
        }
    )
    rows = []
    for x in picked:
        y = dict(x)
        y["probs"] = json.dumps(y["probs"])
        rows.append(y)
    ds = Dataset.from_list(rows, features=feats)
    ds.to_parquet(out_parquet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(getattr(cfg, "random_seed", 42))

    abst = getattr(cfg, "abstain_stream", None)
    if abst is None:
        raise RuntimeError("configs/default.yaml is missing an abstain_stream section")

    cats = list(getattr(abst, "categories", []) or [])
    if not cats:
        raise RuntimeError("abstain_stream.categories must not be empty")
    buf = int(getattr(abst, "hf_shuffle_buffer", 20000))
    buf  = int(getattr(abst, "hf_shuffle_buffer", 20000))
    seed = int(getattr(abst, "seed", 42))
    min_words = int(getattr(abst, "sample_text_min_words", 0))
    dedup = bool(getattr(abst, "dedup", True))

    # Runtime controls
    stop_when_full = bool(getattr(abst, "stop_when_full", False))
    patience_batches = int(getattr(abst, "early_stop_patience_batches", 0))
    max_scanned_total = int(getattr(abst, "max_scanned_total", 0))
    max_scanned_per_category = int(getattr(abst, "max_scanned_per_category", 0))
    max_seconds = int(getattr(abst, "max_seconds", 0))
    write_interval_batches = int(getattr(abst, "write_interval_batches", 0))

    model_dir = getattr(abst, "model_dir", "")
    max_length = int(getattr(abst, "max_length", 256))
    bs = int(getattr(abst, "batch_size", 64))

    topk_total = int(getattr(abst, "topk_total", 0))
    per_pred_quota = str(getattr(abst, "per_pred_quota", "") or "").strip()
    out_jsonl = getattr(abst, "out_jsonl", "data/abstain/to_label.jsonl")
    out_parquet = getattr(abst, "out_parquet", "data/abstain/to_label.parquet")
    seen_sink = getattr(abst, "write_seen_to", "")

    # blocklist
    block_ids = maybe_load_blocklist(getattr(cfg, "blocklist", None))

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    # Selector
    if per_pred_quota and per_pred_quota != "0":
        q = [int(x) for x in per_pred_quota.split(",")]
        if len(q) != 3:
            raise ValueError("per_pred_quota must contain three integers, e.g. 15000,15000,20000")
        selector = PerClassQuota(q)
        per_class_mode = True
    else:
        if topk_total <= 0:
            raise ValueError("Configure either topk_total or per_pred_quota")
        selector = TopK(topk_total)
        per_class_mode = False

    # Streaming + filtering
    seen = set()
    scanned_total = 0
    scanned_per_cat = {c: 0 for c in cats}

    def accept_row(row):
        txt = row["text"]
        if min_words and _wc(txt) < min_words:
            return False, None
        rid = stable_id(txt, row["asin"])
        if rid in block_ids:
            return False, None
        if dedup and rid in seen:
            return False, None
        return True, rid

    # Batch buffers
    batch_texts, batch_meta = [], []
    last_improve_batches = 0  # Number of consecutive batches without improvement
    start_t = time.monotonic()
    batch_idx = 0

    def flush_batch() -> int:
        """Return the number of selections inserted/replaced in this batch."""
        if not batch_texts:
            return 0
        with torch.no_grad():
            enc = tok(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = mdl(**enc).logits
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = prob.argmax(axis=1)
        improved = 0
        for i, meta in enumerate(batch_meta):
            p = prob[i]
            y = int(pred[i])
            score = _uncertainty(p)
            payload = {
                **meta,
                "pred_label": y,
                "probs": p.tolist(),
                "uncertainty": float(score),
            }
            if per_class_mode:
                if selector.push(y, score, payload):
                    improved += 1
            else:
                if selector.push(score, payload):
                    improved += 1
        batch_texts.clear()
        batch_meta.clear()
        return improved

    # Main loop
    for row in _iter_hf_amazon_stream(cats, seed, buf):
        cat = row["product_category"]
        if max_scanned_per_category and scanned_per_cat[cat] >= max_scanned_per_category:
            # Enforce a hard cap per category
            continue

        ok, rid = accept_row(row)
        if not ok:
            continue
        if dedup:
            seen.add(rid)

        scanned_total += 1
        scanned_per_cat[cat] += 1

        meta = {
            "id": rid,
            "text": row["text"],
            "title": row["title"],
            "asin": row["asin"],
            "product_category": cat,
            "rating": float(row.get("rating", 0.0) or 0.0),
        }
        batch_texts.append(row["text"])
        batch_meta.append(meta)

        if len(batch_texts) >= bs:
            batch_idx += 1
            improved = flush_batch()

            # Periodic partial dump for monitoring
            if write_interval_batches and (batch_idx % write_interval_batches == 0):
                picked_partial = selector.result()
                _save_outputs(
                    picked_partial,
                    out_jsonl.replace(".jsonl", ".partial.jsonl"),
                    out_parquet.replace(".parquet", ".partial.parquet"),
                )

            # Early stop tracking
            if stop_when_full and per_class_mode and selector.all_full():
                if improved == 0:
                    last_improve_batches += 1
                else:
                    last_improve_batches = 0
                if patience_batches and last_improve_batches >= patience_batches:
                    print(
                        "[early-stop] all bins full & no improvement for "
                        f"{patience_batches} batches"
                    )
                    break

            # Hard stop checks
            if max_scanned_total and scanned_total >= max_scanned_total:
                print(f"[stop] reached max_scanned_total={max_scanned_total}")
                break
            if max_seconds and (time.monotonic() - start_t) >= max_seconds:
                print(f"[stop] reached max_seconds={max_seconds}")
                break

    # Flush the remainder
    if batch_texts:
        flush_batch()

    picked = selector.result()
    _save_outputs(picked, out_jsonl, out_parquet)

    # Optional stable-id sink
    if seen_sink:
        append_ids(seen_sink, (x["id"] for x in picked))

    print(
        "[done] scanned_total=%s, picked=%s, out=%s, parquet=%s"
        % (scanned_total, len(picked), out_jsonl, out_parquet)
    )


if __name__ == "__main__":
    main()
