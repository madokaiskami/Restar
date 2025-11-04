"""Utilities for mapping review ratings to coarse sentiment labels."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

LABEL_TEXT = {0: "negative", 1: "neutral", 2: "positive"}

__all__ = ["rating_to_class", "class_to_text", "postprocess_logits"]


def rating_to_class(rating: float) -> int:
    """Convert a star rating (1-5) into a sentiment class label."""

    if rating is None:
        raise ValueError("rating is None")
    if not (1.0 <= float(rating) <= 5.0):
        raise ValueError(f"rating out of range: {rating}")

    if rating <= 2:
        return 0
    if rating >= 4:
        return 2
    return 1


def class_to_text(class_id: int) -> str:
    """Return the human readable label for ``class_id``."""

    if class_id not in LABEL_TEXT:
        raise ValueError(f"invalid class id: {class_id}")
    return LABEL_TEXT[class_id]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax implementation for 2-D logits."""

    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D array")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def postprocess_logits(
    logits: np.ndarray,
    threshold: float = 0.65,
) -> Tuple[List[str], List[float]]:
    """Convert model logits to sentiment labels with an abstention option."""

    probs = _softmax(logits)
    confidences = probs.max(axis=-1).tolist()
    predictions = probs.argmax(axis=-1).tolist()

    labels: List[str] = []
    for cls_idx, conf in zip(predictions, confidences):
        if conf < threshold:
            labels.append("abstain")
        else:
            labels.append(LABEL_TEXT[cls_idx])

    return labels, confidences
