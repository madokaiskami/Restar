import argparse
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import load_config
from .labeling import postprocess_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", required=True, help="input review text")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = os.path.join(cfg.output_dir, cfg.run_name, "model")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tok(args.text, return_tensors="pt", truncation=True)
    logits = model(**inputs).logits.detach().numpy()
    labels, conf = postprocess_logits(
        logits,
        threshold=cfg.inference.abstain_threshold,
    )
    print({"label": labels[0], "confidence": conf[0]})


if __name__ == "__main__":
    main()
