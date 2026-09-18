#!/usr/bin/env bash

select_gpu() {
    local gpu_status gpu_index memory_used memory_used_decimal
    local selected_gpu="" minimum_used=""

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf 'Error: nvidia-smi is unavailable.\n' >&2
        return 1
    fi

    if ! gpu_status="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"; then
        printf 'Error: could not query GPU memory usage.\n' >&2
        return 1
    fi

    if [[ -z ${gpu_status//[[:space:]]/} ]]; then
        printf 'Error: no GPUs were detected.\n' >&2
        return 1
    fi

    while IFS=, read -r gpu_index memory_used; do
        gpu_index="${gpu_index//[[:space:]]/}"
        memory_used="${memory_used//[[:space:]]/}"
        if [[ ! $gpu_index =~ ^[0-9]+$ || ! $memory_used =~ ^[0-9]+$ ]]; then
            printf 'Error: invalid GPU memory usage from nvidia-smi.\n' >&2
            return 1
        fi

        memory_used_decimal=$((10#$memory_used))
        if [[ -z $selected_gpu ]] || ((memory_used_decimal < minimum_used)); then
            selected_gpu="$gpu_index"
            minimum_used="$memory_used_decimal"
        fi
    done <<< "$gpu_status"

    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    printf 'Selected GPU: %s\n' "$selected_gpu"
}

select_gpu
