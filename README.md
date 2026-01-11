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
* **Upstream source:** `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face.

### Experiment Plan
1. Materialize the DVC snapshot once (`prepare`) and reuse it for all experiments.
2. **Smoke** run: `prajjwal1/bert-tiny` + fast training parameters + optional row truncation for
   quick end-to-end validation.
3. **Full** run: `distilbert-base-uncased` + complete training parameters for final results.
4. Track model metrics, latency SLAs, and abstention thresholds for release readiness.

* **Reproducible pipelines** with **DVC** (data + models are versioned outside Git).
* **Experiment tracking** with **MLflow** (params/metrics/artifacts).
* **Offline inference** via **Docker** (`ml-app:v1`).
* **Online inference** via **TorchServe** (`mymodel-serve:v1`).
* **CI** (lint/tests + synthetic training + MAR export + Docker builds).

---

## Repository layout

```
.
├─ src/restar/                     # Python package (train / prepare / predict / evaluate)
│  ├─ prepare.py                   # data preparation (supports local_parquet sampling)
│  ├─ train.py                     # training + MLflow logging
│  ├─ evaluate_dataset.py          # evaluation + confusion matrix
│  ├─ predict.py                   # offline batch inference for Docker
│  └─ ...
├─ configs/
│  ├─ dvc_smoke.yaml               # smoke run (small sampling from frozen parquet)
│  ├─ dvc_full.yaml                # full run (larger / real data flow)
│  ├─ data_frozen.yaml             # builds frozen parquet snapshots
│  └─ ci_smoke.yaml                # CI-only synthetic dataset (no DVC, no remote data)
├─ dvc.yaml                        # DVC pipeline definition (prepare/train/evaluate/export)
├─ dvc.lock                        # locked versions of deps/outs (the “truth” for reproducibility)
├─ params.yaml                     # pipeline parameter (restar.dvc_config)
├─ data/
│  ├─ frozen/                      # DVC-tracked frozen datasets (train/dev/eval parquet)
│  └─ raw/                         # DVC-tracked prepared datasets (train/dev/eval parquet)
├─ outputs/dvc_run/                # DVC-tracked training outputs (model + metrics + plots)
│  ├─ model/                       # saved HF model directory
│  ├─ metrics_dev.json
│  ├─ metrics_eval.json
│  └─ confusion_matrix.png
├─ artifacts/pretrained/model/     # DVC-tracked pretrained HF model snapshot
├─ tools/
│  ├─ export_torchserve.py         # produces torchserve/artifacts/model.pt + model-store/mymodel.mar
│  └─ bench_infer.py               # simple inference benchmark
├─ torchserve/
│  ├─ Dockerfile                   # TorchServe container build
│  ├─ handler.py                   # custom handler (pre/post-processing)
│  ├─ config.properties            # TorchServe config
│  ├─ artifacts/                   # exported tokenizer/config files + model.pt (DVC output)
│  └─ model-store/                 # mymodel.mar (DVC output)
├─ Dockerfile                      # offline inference image build (ml-app:v1)
└─ .github/workflows/ci.yml        # CI: lint/tests + synthetic train + mar export + docker builds
```

---

## What is versioned by DVC

Large artifacts are **not stored in Git**, but **are versioned by DVC** (and can be restored via `dvc pull`):

* Frozen datasets:

  * `data/frozen/train.parquet`
  * `data/frozen/dev.parquet`
  * `data/frozen/eval.parquet`
* Prepared datasets:

  * `data/raw/train.parquet`
  * `data/raw/dev.parquet`
  * `data/raw/eval.parquet`
* Pretrained snapshot:

  * `artifacts/pretrained/model/`
* Training outputs:

  * `outputs/dvc_run/model/`
  * `outputs/dvc_run/metrics_dev.json`
  * `outputs/dvc_run/metrics_eval.json`
  * `outputs/dvc_run/confusion_matrix.png`
* TorchServe artifacts:

  * `torchserve/artifacts/model.pt`
  * `torchserve/model-store/mymodel.mar`

The DVC remote is configured in `.dvc/config` (default remote: `storage`).

---

## Setup

### Python environment (training + tools)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e .
pip install -r requirements-dev.txt
```

### DVC (remote uses Google Drive)

```bash
pip install "dvc[gdrive]"
dvc remote list
```

> First `dvc pull` may require Google Drive auth in your browser.

---

## Quick acceptance run (from `git clone` to full verification)

### 1) Clone + install + fetch DVC artifacts

```bash
git clone <REPO_URL>
cd Restar-reserved-main

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
pip install -r requirements-dev.txt
pip install "dvc[gdrive]"

# Pull frozen datasets + pretrained snapshot (recommended before smoke)
dvc pull data/frozen/train.parquet data/frozen/dev.parquet data/frozen/eval.parquet
dvc pull artifacts/pretrained/model
```

### 2) Run the DVC pipeline (smoke)

This repo uses `params.yaml` key `restar.dvc_config` to select a config.

* Default in `params.yaml` is typically `configs/dvc_smoke.yaml`.
* If you want an override without editing files, use DVC experiments:

```bash
# Run with current params.yaml
dvc repro --pull

# OR override config (recommended on older DVC versions)
dvc exp run --pull -S restar.dvc_config=configs/dvc_smoke.yaml
```

Expected outputs:

* `outputs/dvc_run/model/`
* `outputs/dvc_run/metrics_dev.json`
* `outputs/dvc_run/metrics_eval.json`
* `outputs/dvc_run/confusion_matrix.png`
* `torchserve/model-store/mymodel.mar` (if export stage ran)

---

## DVC pipeline stages

`dvc.yaml` defines the pipeline:

1. `fetch_model`
   Downloads/snapshots pretrained model into `artifacts/pretrained/model/`.

2. `freeze_data`
   Builds frozen parquet snapshots in `data/frozen/*.parquet` from `configs/data_frozen.yaml`.

3. `prepare`
   Builds `data/raw/*.parquet`. For smoke, it reads from `data/frozen/*.parquet` and samples deterministically.

4. `train`
   Trains model to `outputs/dvc_run/model/`, logs `outputs/dvc_run/train.log` and dev metrics.

5. `evaluate`
   Writes `outputs/dvc_run/metrics_eval.json` and `outputs/dvc_run/confusion_matrix.png`.

6. `export_torchserve`
   Produces TorchServe artifacts:

   * `torchserve/artifacts/model.pt`
   * `torchserve/model-store/mymodel.mar`

---

## MLflow experiment tracking

### Default behavior

* If `MLFLOW_TRACKING_URI` is not set, MLflow uses the default **local file store** (typically `./mlruns/`).

### Run training with MLflow enabled (local)

```bash
python -m restar.train --config configs/dvc_smoke.yaml --verbose
```

### Start a local MLflow server (recommended for UI)

**Linux/macOS**

```bash
bash scripts/mlflow_server.sh
```

**Windows PowerShell**

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root "$PWD\mlruns" `
  --host 127.0.0.1 `
  --port 5000
```

Then point training to the server:

```bash
# Linux/macOS
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=restar
python -m restar.train --config configs/dvc_smoke.yaml --verbose
```

```powershell
# Windows PowerShell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME="restar"
python -m restar.train --config configs/dvc_smoke.yaml --verbose
```

Open UI: `http://127.0.0.1:5000`

Artifacts logged include model directory and additional run artifacts (e.g., plots / lockfile where applicable).

---

## Offline inference (local + Docker)

### Local (no Docker)

`restar.predict` expects a CSV with a `text` column and outputs a CSV with:
`label, confidence, p_neg, p_neu, p_pos`

```bash
python -m restar.predict \
  --input_path sample_input.csv \
  --output_path preds.csv \
  --model_dir outputs/dvc_run/model
```

### Docker: build & run (ml-app:v1)

**Important:** the Dockerfile copies `outputs/dvc_run/model/`. Make sure it exists:

```bash
dvc pull outputs/dvc_run/model
```

Build:

```bash
docker build -t ml-app:v1 .
```

Run:

```bash
docker run --rm \
  -v "$(pwd)/tmp:/data" \
  ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/preds.csv
```

**Windows PowerShell run example**

```powershell
New-Item -ItemType Directory -Force .\tmp | Out-Null
@"
text
hello world
this is a test
"@ | Set-Content -Encoding UTF8 .\tmp\input.csv

dvc pull outputs/dvc_run/model
docker build -t ml-app:v1 .

docker run --rm `
  -v "${PWD}\tmp:/data" `
  ml-app:v1 `
  --input_path /data/input.csv `
  --output_path /data/preds.csv

Get-Content .\tmp\preds.csv -TotalCount 5
```

---

## TorchServe online service (Docker)

### 1) Ensure `mymodel.mar` exists

Option A (preferred): pull from DVC remote

```bash
dvc pull torchserve/model-store/mymodel.mar
```

Option B: generate from pipeline

```bash
dvc repro export_torchserve
```

### 2) Build TorchServe image

```bash
docker build -t mymodel-serve:v1 -f torchserve/Dockerfile .
```

### 3) Run TorchServe

```bash
docker run -d --rm \
  -p 8080:8080 -p 8081:8081 \
  --name mymodel_serve \
  mymodel-serve:v1
```

### 4) Send a request

The handler accepts JSON with either:

* `{"text": "..."}` (single)
* `{"texts": ["...", "..."]}` (batch)

Example:

```bash
curl -X POST http://127.0.0.1:8080/predictions/mymodel -T sample_input.json
```

Stop:

```bash
docker stop mymodel_serve
```

---

## CI (GitHub Actions) and local parity

CI workflow: `.github/workflows/ci.yml`

* Lint: `ruff check src tests`
* Tests: `pytest -q`
* Synthetic smoke training: `python -m restar.train --config configs/ci_smoke.yaml --verbose`
* Export MAR: `python tools/export_torchserve.py ...`
* Bench: `python tools/bench_infer.py ...`
* Upload artifacts: `outputs/dvc_run/model/**` + `torchserve/model-store/mymodel.mar`
* Separate job downloads artifacts and builds Docker images (offline + TorchServe)

### Run CI steps locally (Linux/macOS)

```bash
pip install -r requirements-dev.txt
pip install ruff

ruff check src tests
PYTHONPATH=src pytest -q

PYTHONPATH=src python -m restar.train --config configs/ci_smoke.yaml --verbose
PYTHONPATH=src python tools/export_torchserve.py \
  --model_dir outputs/dvc_run/model \
  --artifacts_dir torchserve/artifacts \
  --model_store torchserve/model-store \
  --handler torchserve/handler.py
PYTHONPATH=src python tools/bench_infer.py \
  --model-dir outputs/dvc_run/model \
  --warmup 0 \
  --batch-size 2 \
  --output outputs/smoke_ci/bench.json

docker build -t ml-app:v1 .
docker build -t mymodel-serve:v1 -f torchserve/Dockerfile .
```

### Run CI steps locally (Windows PowerShell)

```powershell
pip install -r requirements-dev.txt
pip install ruff

ruff check src tests
$env:PYTHONPATH="src"
pytest -q

python -m restar.train --config configs/ci_smoke.yaml --verbose

python tools/export_torchserve.py `
  --model_dir outputs/dvc_run/model `
  --artifacts_dir torchserve/artifacts `
  --model_store torchserve/model-store `
  --handler torchserve/handler.py

python tools/bench_infer.py `
  --model-dir outputs/dvc_run/model `
  --warmup 0 `
  --batch-size 2 `
  --output outputs/smoke_ci/bench.json

docker build -t ml-app:v1 .
docker build -t mymodel-serve:v1 -f .\torchserve\Dockerfile .
```

---

## Versioning smoke vs full models with Git tags (recommended workflow)

DVC versions the actual binaries; Git tags “pin” the corresponding `dvc.lock` state.

### 1) Run smoke → commit → tag

```bash
dvc exp run --pull -S restar.dvc_config=configs/dvc_smoke.yaml
dvc exp show
dvc exp apply <EXP_REV>

git add dvc.lock params.yaml dvc.yaml configs/dvc_smoke.yaml
git commit -m "train: smoke pipeline (configs/dvc_smoke.yaml)"
git tag -a model-smoke-v1 -m "Smoke model v1"
git push origin HEAD --tags
dvc push
```

### 2) Run full → commit → tag

```bash
dvc exp run --pull -S restar.dvc_config=configs/dvc_full.yaml
dvc exp show
dvc exp apply <EXP_REV>

git add dvc.lock params.yaml dvc.yaml configs/dvc_full.yaml
git commit -m "train: full pipeline (configs/dvc_full.yaml)"
git tag -a model-full-v1 -m "Full model v1"
git push origin HEAD --tags
dvc push
```

### Restore a tagged model later

```bash
git switch --detach model-smoke-v1
dvc pull outputs/dvc_run/model
python -m restar.predict --input_path sample_input.csv --output_path preds_smoke.csv --model_dir outputs/dvc_run/model
```

---

## Troubleshooting notes (practical)

* If `dvc repro` doesn’t support `-S`, use `dvc exp run -S ...`.
* On Windows PowerShell:

  * Use backtick `` ` `` for line continuation (not `\`).
  * Use `curl.exe` (PowerShell `curl` is an alias for `Invoke-WebRequest`).
* Docker builds require model/MAR files to exist locally:

  * `dvc pull outputs/dvc_run/model`
  * `dvc pull torchserve/model-store/mymodel.mar` (or `dvc repro export_torchserve`)
