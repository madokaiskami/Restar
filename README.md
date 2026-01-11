# Re*Star: Reliable Sentiment Re-Rating for Product Reviews

Re*Star is a production-style machine learning project that "re-stars" Amazon review text. The
model predicts the textual sentiment (negative / neutral / positive) for a review and optionally
abstains when the confidence is too low. The repository contains everything required to reproduce
data preparation, model training, evaluation, and TorchServe packaging.

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
  * Average latency ≤ 200 ms for batch size 1 on CPU inference.
  * Failed request ratio ≤ 1 %.
  * Memory/CPU usage within deployment SLA.
* **Model quality:** Accuracy ≥ 90 % on the held-out evaluation split.
* **Abstention policy:** Return `abstain` when the maximum softmax probability is below 0.65.

### Dataset
* **Snapshot (DVC-managed):** `data/raw/train.parquet`, `data/raw/dev.parquet`,
  `data/raw/eval.parquet` are the single frozen snapshots used for training/evaluation.
* **Upstream source:** `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face; dev/eval splits are pulled
  from `Aurelianous/restar_v1.0_eval` and materialized into the parquet snapshot via `prepare`.
* **Unified schema:** `text`, `rating`, `title`, `asin`, `product_category`.

### Experiment Plan
1. Materialize the DVC snapshot once (`prepare`) and reuse it for all experiments.
2. **Smoke** run: `prajjwal1/bert-tiny` + fast training parameters + optional row truncation for
   quick end-to-end validation.
3. **Full** run: `distilbert-base-uncased` + complete training parameters for final results.
4. Track model metrics, latency SLAs, and abstention thresholds for release readiness.

## Repository Structure
```
Restar-reserved/
├─ configs/
│  ├─ dvc_smoke.yaml         # Small model + fast training parameters
│  ├─ dvc_full.yaml          # Full training configuration
│  └─ legacy/                # Older configs retained for reference
├─ params.yaml               # Active DVC config selector
├─ src/restar/
│  ├─ prepare.py             # DVC prepare stage
│  ├─ train.py               # Training entry point (Trainer + save_pretrained)
│  ├─ evaluate_dataset.py    # Evaluation stage
│  ├─ fetch_pretrained.py    # Fetch HF checkpoint for offline runs
│  └─ tools/                 # Non-critical utilities (snapshot, abstain, infer)
├─ tools/                    # Auxiliary scripts (HF push, blocklist, TorchServe export)
├─ torchserve/               # Handler + Dockerfile for TorchServe
├─ tests/                    # Unit tests for preprocessing and training glue
└─ README.md
```

## Quick Start
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Smoke training (fast)
python -m restar.train --config configs/dvc_smoke.yaml --verbose

# Full training (larger model)
python -m restar.train --config configs/dvc_full.yaml --verbose

# Benchmark CPU inference latency (writes outputs/bench.json)
python tools/bench_infer.py --model-dir outputs/dvc_run/model
```

The resulting model checkpoints are saved to `outputs/<run_name>/model/` using the Hugging Face
`save_pretrained()` format.

## DVC Pipeline (Task 1)
All reproducible assets (data, pretrained checkpoints, model outputs, TorchServe artifacts) are
versioned through DVC.

**Tracked locations**
* Raw data snapshot: `data/raw/train.parquet`, `data/raw/dev.parquet`, `data/raw/eval.parquet`.
* Pretrained snapshot materialized for offline training: `artifacts/pretrained/model/`.
* Training outputs: `outputs/dvc_run/model`, `outputs/dvc_run/metrics_*.json`, `outputs/dvc_run/train.log`.
* TorchServe artifacts: `torchserve/artifacts/model.pt`, `torchserve/model-store/mymodel.mar`.

**Default smoke run**
```bash
git clone <repo-url>
cd Restar-reserved
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-build-isolation
python -m pip install "dvc[gdrive]"  # first-time dependency

dvc pull  # fetch snapshot + pretrained artifacts (if remote configured)
dvc repro  # runs fetch_model -> prepare -> train -> evaluate -> export_torchserve
```

**Full configuration (experiment)**
```bash
dvc exp run -S restar.dvc_config=configs/dvc_full.yaml
```

**MLflow UI**
```bash
mlflow ui --backend-store-uri file:./mlruns
```

If you see missing assets, run `dvc pull` to hydrate the snapshot before rerunning the pipeline.

## Docker Offline Inference (Task 3)
Build the offline inference image after fetching the DVC-tracked model artifacts:

```bash
dvc pull
docker build -t ml-app:v1 .
```

Run the container with a CSV input and output volume mounted:

```bash
docker run --rm \
  -v "$(pwd)":/app/workspace \
  ml-app:v1 \
  --input_path /app/workspace/sample_input.csv \
  --output_path /app/workspace/preds.csv
```

### Input CSV schema
* Required column: `text` (string)

### Output CSV schema
* `label`: `negative`, `neutral`, `positive`, or `abstain`
* `confidence`: maximum softmax probability
* `p_neg`, `p_neu`, `p_pos`: class probabilities

### Logging and Metrics
* Console logs plus `outputs/<run_name>/train.log` (configured through the Python `logging` module).
* Final metrics include accuracy, macro F1, and the confusion matrix for dev/eval splits.

### Reproducibility
* `random_seed` is applied to Python, NumPy, and PyTorch.
* All tunable parameters—data locations, sampling quotas, optimization settings—are defined in YAML
  configs for deterministic reruns.

## TorchServe Deployment (Task 4)
This directory ships a TorchServe-ready handler and packaging scripts without affecting the offline
inference Dockerfile.

### Export TorchServe artifacts
The DVC stage `export_torchserve` runs the exporter, which writes `torchserve/artifacts/model.pt`
plus tokenizer/config files and builds `torchserve/model-store/mymodel.mar`.

```bash
python tools/export_torchserve.py \
  --model_dir outputs/dvc_run/model \
  --artifacts_dir torchserve/artifacts \
  --model_store torchserve/model-store \
  --handler torchserve/handler.py
```

> Note: `torch-model-archiver` is required (included in `requirements.txt`).

### TorchServe Docker image
Build and run the TorchServe container:

```bash
docker build -t mymodel-serve:v1 -f torchserve/Dockerfile .
```

```bash
docker run -d -p 8080:8080 -p 8081:8081 mymodel-serve:v1
```

### Inference examples
Single input:

```bash
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Content-Type: application/json" \
  -d '{"text": "The vacuum works but the battery is weak."}'
```

Batch input:

```bash
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Love the screen.", "Battery life is mediocre."]}'
```

### JSON schema
**Request**
```json
{"text": "string"}
```
or
```json
{"texts": ["string", "string"]}
```

**Response**
```json
[
  {
    "label": "negative|neutral|positive",
    "confidence": 0.0,
    "probs": [0.0, 0.0, 0.0]
  }
]
```

## Data Validation & Testing
* The data loader validates schema, types, rating range, and text lengths.
* Tests cover preprocessing, label mapping, leakage guards between train/eval IDs, configuration
  parsing, and a synthetic end-to-end training smoke test.
* GitHub Actions (see `.github/workflows/ci.yml`) runs Ruff linting, `pytest`, the smoke training
  config, and a CPU inference benchmark on every push.
* CI also builds the offline inference Docker image (`ml-app:v1`) and the TorchServe image
  (`mymodel-serve:v1`) to ensure the Dockerfiles stay green.

### Latency Benchmark Snapshot

Run `tools/bench_infer.py` after training to track SLA compliance. The script measures single and
batch inference on CPU and exports metrics to `outputs/bench.json`.

## Human-in-the-Loop Loop
1. Stream snapshots with `restar.tools.snapshot_stream` to create raw JSONL shards.
2. Use `tools/split_and_push_to_hf.py` to build stratified dev/eval splits and optionally push to
   Hugging Face datasets.
3. Run `restar.tools.abstain_stream` to gather the least confident predictions from the model.
4. Label candidates via `tools/label_to_parquet.py` and merge them into curated training data.
5. Retrain with updated weights or instance-level sampling to continuously improve recall on hard
   cases.

## License & Acknowledgements
* Original review data © the respective platform owners.
* Code and model checkpoints are released under the Apache-2.0 license.
