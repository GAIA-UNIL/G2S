#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_ROOT="${SCRIPT_DIR}/algorithms/example"

mkdir -p "${TARGET_ROOT}/python" "${TARGET_ROOT}/matlab"

rm -rf "${TARGET_ROOT}/python" "${TARGET_ROOT}/matlab"
mkdir -p "${TARGET_ROOT}/python" "${TARGET_ROOT}/matlab"

cp -R "${REPO_ROOT}/example/python/." "${TARGET_ROOT}/python/"
cp -R "${REPO_ROOT}/example/matlab/." "${TARGET_ROOT}/matlab/"
find "${TARGET_ROOT}" -type d -name "__pycache__" -prune -exec rm -rf {} +
