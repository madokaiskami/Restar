"""Shared evaluation helpers for reporting metrics and artifacts."""

from __future__ import annotations

import os
from typing import Any, Dict


def maybe_plot_confusion(metrics: Dict[str, Any], out_path: str) -> None:
    cm = (
        metrics.get("eval_confusion_matrix") or metrics.get("confusion_matrix") or metrics.get("cm")
    )
    if not cm:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception:
        return

    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    for (i, j), val in np.ndenumerate(arr):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
