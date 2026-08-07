#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${script_directory}/.." && pwd)"
environment_directory="${project_directory}/.venv"
model_directory="${project_directory}/models"
model_path="${model_directory}/holistic_landmarker.task"
model_url="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

python_command="${AGAPE_PYTHON:-}"
if [[ -z "${python_command}" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c \
      'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))'; then
      python_command="${candidate}"
      break
    fi
  done
fi
if [[ -z "${python_command}" ]]; then
  echo "AGAPE requires Python 3.11, 3.12, or 3.13." >&2
  exit 1
fi
if ! command -v "${python_command}" >/dev/null 2>&1 || ! "${python_command}" -c \
  'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))'; then
  echo "AGAPE_PYTHON must resolve to Python 3.11, 3.12, or 3.13." >&2
  exit 1
fi

for dependency in ffmpeg ffprobe curl; do
  if ! command -v "${dependency}" >/dev/null 2>&1; then
    echo "Missing required dependency: ${dependency}" >&2
    exit 1
  fi
done

"${python_command}" -m venv "${environment_directory}"
"${environment_directory}/bin/python" -m pip install --upgrade pip
"${environment_directory}/bin/python" -m pip install -e "${project_directory}[dev]"

mkdir -p "${model_directory}"
if [[ ! -f "${model_path}" ]]; then
  curl --fail --location --output "${model_path}" "${model_url}"
fi

echo "AGAPE is ready."
echo "Activate it with: source ${environment_directory}/bin/activate"
