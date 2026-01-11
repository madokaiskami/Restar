"""Batch CSV inference for offline Docker usage."""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def run_prediction(
    input_path: str,
    output_path: str,
    model_dir: str,
    batch_size: int,
    abstain_threshold: float,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    texts = df["text"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    probs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    stacked = np.vstack(probs) if probs else np.zeros((0, 3), dtype=np.float32)
    if len(texts) != len(stacked):
        raise ValueError("Mismatch between inputs and prediction outputs")

    p_neg = stacked[:, 0]
    p_neu = stacked[:, 1]
    p_pos = stacked[:, 2]
    max_probs = stacked.max(axis=1) if stacked.size else np.array([])
    max_labels = stacked.argmax(axis=1) if stacked.size else np.array([], dtype=int)

    labels = [LABEL_MAP.get(int(idx), "unknown") for idx in max_labels]
    labels = [
        "abstain" if prob < abstain_threshold else label
        for label, prob in zip(labels, max_probs, strict=False)
    ]

    output_df = pd.DataFrame(
        {
            "label": labels,
            "confidence": max_probs,
            "p_neg": p_neg,
            "p_neu": p_neu,
            "p_pos": p_pos,
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"[ok] wrote {len(output_df)} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_dir", default="outputs/dvc_run/model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--abstain_threshold", type=float, default=0.0)
    args = parser.parse_args()

    run_prediction(
        input_path=args.input_path,
        output_path=args.output_path,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        abstain_threshold=args.abstain_threshold,
    )


if __name__ == "__main__":
    main()
