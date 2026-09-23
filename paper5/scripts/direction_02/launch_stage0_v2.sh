#!/usr/bin/env bash
# One complete V2 attempt, fixed physical GPU2 and the inherited total budget.
set -euo pipefail
[[ "$(git branch --show-current)" == paper5 ]]
[[ "$(git rev-parse HEAD)" == "${D2_EXPECTED_SHA:?exact committed SHA required}" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
: "${D2_PYTHON:?checked D2 interpreter required}"
: "${D2_MODEL_SNAPSHOT:?pinned snapshot required}"
: "${D2_CACHE:?D2 cache parent required}"
: "${D2_GPU_PERMISSION_FILE:?V2 shared permission required}"
: "${D2_V2_LOG_DIR:?independent V2 project scratch directory required}"
umask 077
mkdir -p "$(dirname "$D2_V2_LOG_DIR")"
mkdir "$D2_V2_LOG_DIR"
exec > "$D2_V2_LOG_DIR/run.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$D2_V2_LOG_DIR/exit_code"' EXIT
printf '%s\n' "$$" > "$D2_V2_LOG_DIR/launcher_pid"
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$D2_CACHE/recovery_stage0_v2/huggingface"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export D2_GPU_PHASE_STARTED_UNIX
D2_GPU_PHASE_STARTED_UNIX=$(date +%s)
remaining_seconds=$("$D2_PYTHON" -c 'import math; from paper5.experiments.canary.direction_02 import run_stage0 as r, stage0_quality_v2 as v; v.activate(r); print(math.floor(r.CFG["gpu_seconds_limit"] - r.data.read_json(r.OUT / "prior_gpu_budget.json")["spent_seconds"] - 5))')
[[ "$remaining_seconds" -gt 30 ]]
timeout --signal=TERM --kill-after=5s "$remaining_seconds" "$D2_PYTHON" -m paper5.experiments.canary.direction_02.run_stage0 --quality-v2 --stage formal --snapshot "$D2_MODEL_SNAPSHOT"
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 1795s "$D2_PYTHON" -m paper5.experiments.canary.direction_02.run_stage0 --quality-v2 --stage analyze
