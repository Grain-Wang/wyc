#!/usr/bin/env bash
# Start only from the remote repository, inside one detached screen/tmux session.
set -euo pipefail
stage=${1:?pilot or formal required}
[[ "$stage" == pilot || "$stage" == formal ]]
[[ "$(git branch --show-current)" == paper5 ]]
[[ "$(git rev-parse HEAD)" == "${D2_EXPECTED_SHA:?exact committed code SHA required}" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
: "${D2_PYTHON:?set the checked project interpreter}"
: "${D2_MODEL_SNAPSHOT:?set the existing pinned D1 model snapshot}"
: "${D2_CACHE:?set project D2 cache parent directory}"
: "${D2_GPU_PERMISSION_FILE:?verified server allocation/permission record required}"
run_id=20260923_stage0_01
run_directory="$HOME/whr/paper5/scratch/direction_02/$run_id/$stage"
umask 077
mkdir -p "$(dirname "$run_directory")"
mkdir "$run_directory"
exec > "$run_directory/run.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$run_directory/exit_code"' EXIT
printf '%s\n' "$$" > "$run_directory/launcher_pid"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$D2_CACHE/huggingface"
export D2_STAGE="$stage"
read -r selected_gpu remaining_seconds < <("$D2_PYTHON" -m paper5.experiments.canary.direction_02.stage0_resources)
export CUDA_VISIBLE_DEVICES="$selected_gpu"
printf 'code_sha=%s stage=%s selected_gpu=%s\n' "$D2_EXPECTED_SHA" "$stage" "$selected_gpu"
date --utc --iso-8601=seconds
export D2_GPU_PHASE_STARTED_UNIX
D2_GPU_PHASE_STARTED_UNIX=$(date +%s)
timeout --signal=TERM --kill-after=5s "$remaining_seconds" "$D2_PYTHON" -m paper5.experiments.canary.direction_02.run_stage0 --stage "$stage" --snapshot "$D2_MODEL_SNAPSHOT"
if [[ "$stage" == formal ]]; then
    CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 1795s "$D2_PYTHON" -m paper5.experiments.canary.direction_02.run_stage0 --stage analyze
fi
date --utc --iso-8601=seconds
