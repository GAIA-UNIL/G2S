#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_ROOT="${SCRIPT_DIR}/algorithms/example"

mkdir -p "${TARGET_ROOT}/python" "${TARGET_ROOT}/matlab"

find "${TARGET_ROOT}/python" -type f -delete
find "${TARGET_ROOT}/matlab" -type f -delete

cp "${REPO_ROOT}/example/python/unconditionalSimulation.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/conditionalSimulation.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/Multiple_TI.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/multivariateSimulation.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/gapFilling.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/downscaling.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/3D.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/qs_rotation_2d.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/qs_rotation_equivalence_2d.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/qs_scale_2d.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/qs_quaternion_rotation_3d.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/qs_scale_rotation_2d.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/async_mode.py" "${TARGET_ROOT}/python/"
cp "${REPO_ROOT}/example/python/AnchorSampling.py" "${TARGET_ROOT}/python/"

cp "${REPO_ROOT}/example/matlab/unconditionalSimulation.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/conditionalSimulation.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/Multiple_TI.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/multivariateSimulation.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/gapFilling.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/downscaling.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/3D.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/async_mode.m" "${TARGET_ROOT}/matlab/"
cp "${REPO_ROOT}/example/matlab/AnchorSampling.m" "${TARGET_ROOT}/matlab/"
