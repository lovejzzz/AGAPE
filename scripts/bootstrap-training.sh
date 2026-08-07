#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${script_directory}/.." && pwd)"

"${script_directory}/bootstrap.sh"
"${project_directory}/.venv/bin/python" -m pip install -e "${project_directory}[dev,train]"

mkdir -p "${project_directory}/training_data" "${project_directory}/training_runs"

"${project_directory}/.venv/bin/python" - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
}, indent=2))
PY

echo "AGAPE local training is ready."
echo "Run: .venv/bin/agape train-demo"
