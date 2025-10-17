#!/usr/bin/env bash
set -euo pipefail

# Usage: source ./setup_env.sh  (so the venv activates in current shell)
# Notes:
# - Designed for Python 3.10
# - Respects existing .env (not overwritten)
# - Installs CPU PyTorch by default; see README for GPU options

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="python3.10"

if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
  echo "Python 3.10 not found as ${PYTHON_BIN}. Falling back to python3." >&2
  PYTHON_BIN="python3"
fi

# Create venv if missing
if [ ! -d "${VENV_DIR}" ]; then
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

REQ_FILE="${PROJECT_ROOT}/requirements.txt"
if [ ! -f "${REQ_FILE}" ]; then
  echo "requirements.txt not found at ${REQ_FILE}" >&2
  exit 1
fi

pip install -r "${REQ_FILE}"

# Download common NLTK assets
python - <<'PY'
import nltk
assets = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "vader_lexicon",
]
for pkg in assets:
    try:
        nltk.download(pkg)
    except Exception as e:
        print(f"Failed to download {pkg}: {e}")
PY

# Respect existing .env, otherwise copy example
if [ ! -f "${PROJECT_ROOT}/.env" ] && [ -f "${PROJECT_ROOT}/.env.example" ]; then
  cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
  echo "Created .env from .env.example. Remember to add API keys."
else
  echo ".env present; not modifying."
fi

echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
