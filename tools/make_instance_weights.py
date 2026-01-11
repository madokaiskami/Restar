"""Generate instance-level weights based on model agreement and confidence."""

import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from restar.labeling import rating_to_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--model_dir", default="outputs/distilbert_amazon_restar/model")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    if "label" not in df.columns:
        df["label"] = df["rating"].apply(rating_to_class)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()

    dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)
    probs: List[np.ndarray] = []
    batch_size = 64

    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
            probs.append((exp_logits / exp_logits.sum(dim=1, keepdim=True)).cpu().numpy())

    stacked = np.vstack(probs)
    predictions = stacked.argmax(axis=1)
    gold = df["label"].to_numpy()
    agreement = (predictions == gold).astype(float)
    confidence = stacked.max(axis=1)

    weights = 0.3 + 0.7 * confidence
    weights[agreement == 0] *= 0.5
    if "rating" in df.columns:
        neutral_mask = (df["rating"] == 3).to_numpy()
        weights[neutral_mask] *= 0.8

    df["sample_weight"] = weights
    out_path = args.out_csv or args.train_csv.replace(".csv", "_with_weight.csv")
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote weights -> {out_path}")


if __name__ == "__main__":
    main()
