# Re*Star: Sentiment Re-Rating for Product Reviews

This is a course project of MLops.

Re*Star is a production-style machine learning project that "re-stars" Amazon review text. The
model predicts the textual sentiment (negative / neutral / positive) for a review and optionally
abstains when the confidence is too low. The repository contains everything required to reproduce
data preparation, model training, evaluation, and abstention sampling.

## Business Goal
* **Problem statement.** Marketplace review ratings are often inconsistent with the textual
  content. Operators cannot trust the star score for ranking, moderation, or analytics.
* **Solution.** Train a text classification model that aligns textual sentiment with three target
  classes (≤2 stars, 3 stars, ≥4 stars). Serve predictions together with a calibrated abstention
  option so downstream systems can ignore uncertain results.

### Target Metrics
* **Business:** On downstream ranking A/B tests the bounce rate should drop by at least 2 percentage
  points when using Re*Star predictions (proxy targets are used in this repository).
* **Service SLA:**
  * Average latency ≤ 500 ms for batch size 1 on CPU inference.
  * Failed request ratio ≤ 1 %.
  * Memory/CPU usage within deployment SLA.
* **Model quality:** Accuracy ≥ 85 % on the held-out evaluation split.
* **Abstention policy:** Return `abstain` when the maximum softmax probability is below 0.65.

### Dataset
* Source: `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face.
* Unified schema: `text`, `rating`, `title`, `asin`, `product_category`.
* Filtering: keep texts with 10–512 words and valid ratings; sample uniformly across the configured
  product categories.
* Custom CSV/JSON inputs are supported via configuration.

### Experiment Plan
1. Freeze review snapshots per category.
2. Stratified sampling to build dev/eval splits (pushable to Hugging Face).
3. Train transformer classifiers with loss re-weighting to mitigate class imbalance.
4. Generate abstention pools and collect human labels for low-confidence cases.
5. Iterate by merging new labels, re-training, and monitoring metrics and SLAs.

## Repository Structure
```
Restar/
├─ configs/                 # YAML configs for data, training, and sampling
│  ├─ default.yaml          # Main training setup (DistilBERT)
│  ├─ smoke.yaml            # Synthetic smoke-test configuration used in CI
│  ├─ snapshot_split.yaml   # Snapshot stratification + HF push configuration
│  └─ tiny.yaml             # Additional lightweight configuration for experiments
│  ├─ snapshot.yaml         # Snapshot streaming configuration
│  └─ tiny.yaml             # Lightweight configuration for tests
├─ src/restar/
│  ├─ config.py             # YAML loader with tolerant namespaces
│  ├─ data.py               # Data streaming, validation, and dataset builders
│  ├─ train.py              # Training entry point (Trainer + save_pretrained)
│  ├─ evaluate.py           # Evaluation / inference CLI utilities
│  ├─ abstain_stream.py     # Uncertainty sampling pipeline
│  ├─ snapshot_stream.py    # Snapshot creation from Hugging Face streaming data
│  └─ utils.py              # Utilities (stable IDs, blocklists, seeding)
├─ scripts/
│  ├─ split_and_push_to_hf.py    # Stratified sampling + HF dataset push
│  ├─ label_to_parquet.py        # Interactive labeling tool
│  └─ push_model_to_hf.py        # Convenience helper for model uploads
├─ tests/                   # Unit tests for preprocessing and training glue
├─ outputs/                 # Training artifacts (created at runtime)
├─ pyproject.toml           # Tooling configuration (ruff, packaging metadata)
├─ pytest.ini               # pytest defaults
└─ README.md
```

## Quick Start
```bash
pip install -r requirements.txt
pip install -e .

# Full training run
python -m restar.train --config configs/default.yaml --verbose

# Lightweight smoke test configuration
python -m restar.train --config configs/smoke.yaml --verbose

# Basic evaluation with abstention
python -m restar.evaluate --config configs/tiny.yaml --text "The vacuum works but the battery is weak."
# Benchmark CPU inference latency (writes outputs/bench.json)
python scripts/bench_infer.py --model-dir outputs/restar_v1_0/model
```
The resulting model checkpoints are saved to `outputs/<run_name>/model/` using the Hugging Face
`save_pretrained()` format.

### Logging and Metrics
* Console logs plus `outputs/<run_name>/train.log` (configured through the Python `logging` module).
* Final metrics include accuracy, macro F1, and the confusion matrix for dev/eval splits.

### Reproducibility
* `random_seed` is applied to Python, NumPy, and PyTorch.
* All tunable parameters—data locations, sampling quotas, optimization settings—are defined in YAML
  configs for deterministic reruns.

## Data Validation & Testing
* The data loader validates schema, types, rating range, and text lengths.
* Tests cover preprocessing, label mapping, leakage guards between train/eval IDs, configuration
  parsing, and a synthetic end-to-end training smoke test.
* GitHub Actions (see `.github/workflows/ci.yml`) runs Black formatting, Ruff linting, `pytest`,
  the smoke-training config, and a CPU inference benchmark on every push.

### Latency Benchmark Snapshot

Run `scripts/bench_infer.py` after training to track SLA compliance. The script measures single and
batch inference on CPU and exports metrics to `outputs/bench.json`. Below is a sample JSON payload
captured from the benchmark CLI to document the target latencies:

```json
{
  "timestamp": "2024-06-01T12:00:00Z",
  "model": "outputs/restar_v1_0/model",
  "hardware": "cpu",
  "batch_size_1": {"p50_ms": 132.4, "p95_ms": 181.9},
  "batch_size_8": {"p50_ms": 268.7, "p95_ms": 312.5}
}
```


# Basic evaluation with abstention
python -m restar.evaluate --config configs/tiny.yaml --text "The vacuum works but the battery is weak."
# Benchmark CPU inference latency (writes outputs/bench.json)
python scripts/bench_infer.py --model-dir outputs/restar_v1_0/model
python -m restar.train --config configs/tiny.yaml --verbose

# Basic evaluation with abstention
python -m restar.evaluate --config configs/tiny.yaml --text "The vacuum works but the battery is weak."
```
The resulting model checkpoints are saved to `outputs/<run_name>/model/` using the Hugging Face
`save_pretrained()` format.

### Logging and Metrics
* Console logs plus `outputs/<run_name>/train.log` (configured through the Python `logging` module).
* Final metrics include accuracy, macro F1, and the confusion matrix for dev/eval splits.

### Reproducibility
* `random_seed` is applied to Python, NumPy, and PyTorch.
* All tunable parameters—data locations, sampling quotas, optimization settings—are defined in YAML
  configs for deterministic reruns.

## Data Validation & Testing (Task 3)
* The data loader validates schema, types, rating range, and text lengths.
* Tests cover preprocessing, label mapping, leakage guards between train/eval IDs, configuration
  parsing, and a synthetic end-to-end training smoke test.
* GitHub Actions (see `.github/workflows/ci.yml`) runs Black formatting, Ruff linting, `pytest`,
  the smoke-training config, and a CPU inference benchmark on every push.

### Latency Benchmark Snapshot

Run `scripts/bench_infer.py` after training to track SLA compliance. The script measures single and
batch inference on CPU and exports metrics to `outputs/bench.json`. Below is a sample JSON payload
captured from the benchmark CLI to document the target latencies:

```json
{
  "timestamp": "2024-06-01T12:00:00Z",
  "model": "outputs/restar_v1_0/model",
  "hardware": "cpu",
  "batch_size_1": {"p50_ms": 132.4, "p95_ms": 181.9},
  "batch_size_8": {"p50_ms": 268.7, "p95_ms": 312.5}
}
```
* Tests cover preprocessing, label mapping, configuration parsing, and a tiny end-to-end training
  smoke test.
* GitHub Actions (see `.github/workflows/ci.yml`) runs linting and `pytest` on every push.

## Human-in-the-Loop Loop
1. Stream snapshots with `restar.snapshot_stream` to create raw JSONL shards.
2. Use `scripts/split_and_push_to_hf.py` to build stratified dev/eval splits and optionally push to
   Hugging Face datasets.
3. Run `restar.abstain_stream` to gather the least confident predictions from the model.
4. Label candidates via `scripts/label_to_parquet.py` and merge them into curated training data.
5. Retrain with updated weights or instance-level sampling to continuously improve recall on hard
   cases.

## License & Acknowledgements
* Original review data © the respective platform owners.
* Code and model checkpoints are released under the Apache-2.0 license.
