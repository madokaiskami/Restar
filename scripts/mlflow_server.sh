#!/usr/bin/env bash
set -euo pipefail

BACKEND_STORE_URI="sqlite:///mlflow.db"
ARTIFACT_ROOT="./mlflow_artifacts"

mkdir -p "${ARTIFACT_ROOT}"

echo "MLflow UI: http://127.0.0.1:5000"

mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 127.0.0.1 \
  --port 5000
