"""Benchmark inference latency for the Re*Star model on CPU."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _load_texts(path: str | None, fallback_size: int) -> List[str]:
    if path:
        texts: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                entry = line.strip()
                if not entry:
                    continue
                try:
                    payload = json.loads(entry)
                    text = payload.get("text") if isinstance(payload, dict) else None
                    if text:
                        texts.append(str(text))
                except json.JSONDecodeError:
                    texts.append(entry)
        if texts:
            return texts
    base = [
        "Great sound quality and battery life!",
        "Mediocre performance, nothing special.",
        "Arrived broken and support was unhelpful.",
        "Exactly as described. Works flawlessly.",
        "Disappointed with the durability after a week.",
    ]
    rng = random.Random(42)
    while len(base) < fallback_size:
        base.append(rng.choice(base))
    return base


def _percentiles(latencies_ms: List[float]) -> dict:
    if not latencies_ms:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "count": len(latencies_ms),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


def benchmark(
    model_dir: str,
    input_path: str | None,
    batch_size: int,
    iters: int,
    warmup: int,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    texts = _load_texts(input_path, fallback_size=max(batch_size * 2, 16))
    single_lat: List[float] = []
    batch_lat: List[float] = []

    with torch.inference_mode():
        for idx in range(warmup + iters):
            text = random.choice(texts)
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            start = time.perf_counter()
            model(**encoded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if idx >= warmup:
                single_lat.append(elapsed)

        for idx in range(warmup + iters):
            batch_text = [random.choice(texts) for _ in range(batch_size)]
            encoded = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            start = time.perf_counter()
            model(**encoded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if idx >= warmup:
                batch_lat.append(elapsed)

    return {
        "model_dir": model_dir,
        "device": str(device),
        "batch_size": batch_size,
        "single_sample_ms": _percentiles(single_lat),
        "batch_ms": _percentiles(batch_lat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="outputs/restar_v1_0/model",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional JSONL file with review texts",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="Benchmark iterations per mode",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per mode",
    )
    parser.add_argument(
        "--output",
        default="outputs/bench.json",
        help="Destination for benchmark metrics",
    )
    args = parser.parse_args()

    result = benchmark(args.model_dir, args.input_jsonl, args.batch_size, args.iters, args.warmup)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
