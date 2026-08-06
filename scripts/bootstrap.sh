#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${script_directory}/.." && pwd)"
environment_directory="${project_directory}/.venv"
model_directory="${project_directory}/models"
model_path="${model_directory}/holistic_landmarker.task"
model_url="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

for dependency in python3 ffmpeg ffprobe curl; do
  if ! command -v "${dependency}" >/dev/null 2>&1; then
    echo "Missing required dependency: ${dependency}" >&2
    exit 1
  fi
done

python3 -m venv "${environment_directory}"
"${environment_directory}/bin/python" -m pip install --upgrade pip
"${environment_directory}/bin/python" -m pip install -e "${project_directory}[dev]"

mkdir -p "${model_directory}"
if [[ ! -f "${model_path}" ]]; then
  curl --fail --location --output "${model_path}" "${model_url}"
fi

echo "AGAPE is ready."
echo "Activate it with: source ${environment_directory}/bin/activate"
