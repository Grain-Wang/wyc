#!/usr/bin/env bash
# Invoke once inside detached screen, from the remote repository root.
set -eo pipefail
code_commit=$(git rev-parse HEAD)
run_directory="$HOME/whr/paper5/scratch/h1-multi-edit-${code_commit}"
mkdir "$run_directory"
exec > "$run_directory/run.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_directory/exit_code"' EXIT
printf 'executed_code_commit=%s\nlauncher_pid=%s\n' "$code_commit" "$$"
date --utc --iso-8601=seconds
source "${PAPER5_CONDA_INIT:?Set the authorized Conda initialization script}"
conda activate autoresearch_paper5
set -u
source paper5/scripts/select_gpu.sh
export PAPER5_DATA_ROOT="$HOME/whr/paper5/data"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi
python -c '
import csv, io, json, os, subprocess
from pathlib import Path
selected = os.environ["CUDA_VISIBLE_DEVICES"]
raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
rows = list(csv.reader(io.StringIO(raw)))
row = next(row for row in rows if row[0].strip() == selected)
if "A800" not in row[2] or int(row[4]) < 8192:
    raise RuntimeError("Selected GPU lacks required 4096 MiB peak allowance + 4096 MiB reserve; stop")
out = Path("paper5/results/canary/direction_01/H1_multi_edit/resource_preflight.json")
with out.open("x") as stream:
    json.dump({"selected_gpu":selected,"cuda_visible_devices":selected,"gpu_uuid":row[1].strip(),"gpu_name":row[2].strip(),"memory_used_mib":int(row[3]),"memory_free_mib":int(row[4]),"utilization_percent":int(row[5]),"minimum_free_mib":8192,"no_other_processes_terminated":True}, stream, indent=2)
'
# Outer deadlines include imports and loading; TERM then KILL by the hard limit.
timeout --signal=TERM --kill-after=5s 7195s python -m paper5.experiments.canary.direction_01.run_h1_multi_edit --stage evaluate
CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 1795s python -m paper5.experiments.canary.direction_01.run_h1_multi_edit --stage analyze
date --utc --iso-8601=seconds
