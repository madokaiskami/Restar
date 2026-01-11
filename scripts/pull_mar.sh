#!/usr/bin/env bash
set -euo pipefail

if dvc pull torchserve/model-store/mymodel.mar; then
  exit 0
fi

echo "[pull_mar] TorchServe MAR not available via DVC pull." >&2
echo "[pull_mar] You can generate it by running: dvc repro export_torchserve" >&2
exit 1
