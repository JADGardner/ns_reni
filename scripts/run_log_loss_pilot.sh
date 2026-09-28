#!/usr/bin/env bash
# Isolated csgpu13 pilot. Invoke with ROOT GPU VARIANT NAME [STEPS].
# ROOT contains a frozen source/ snapshot and separate runs/. Never writes thesis files.
set -euo pipefail
pilot_root="$1"
gpu="$2"
variant="$3"
name="$4"
steps="${5:-50001}"
case "$gpu" in 0|1|2) ;; *) exit 2 ;; esac
case "$variant" in legacy|per_image|linear_cosine|both) ;; *) exit 2 ;; esac
[[ "$name" =~ ^[a-z0-9_]+$ ]] || exit 2
run_dir="$pilot_root/runs/$name"
mkdir -p "$pilot_root/runs"
mkdir "$run_dir"
exec >"$run_dir/console.log" 2>&1
printf '%s\n' "$$" >"$run_dir/worker.pid"
finish() {
    code=$?
    printf '%s\n' "$code" >"$run_dir/exit_code"
    date -u +%FT%TZ >"$run_dir/finished_utc"
}
trap finish EXIT
date -u +%FT%TZ >"$run_dir/started_utc"
export CONTAINER_DIR=/home/userfs/j/jadg502/Personal_Staffstore/phd/.apptainer
export DATA_PATH=/home/userfs/j/jadg502/Personal_Staffstore/data
export MODEL_STORAGE_PATH=/home/userfs/j/jadg502/Personal_Staffstore/model-storage
export OUTPUTS_PATH="$pilot_root/runs"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu"
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_WANDB_MODE=offline
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TORCH_HOME=/workspace/outputs/_cache/torch
export APPTAINERENV_MPLCONFIGDIR=/workspace/outputs/_cache/matplotlib
cd "$pilot_root/source"
bash .apptainer/apptainer.sh exec --ro -- env PYTHONPATH=/workspace/code:/overlay-packages \
    python /workspace/code/scripts/train_reni.py \
    --data /workspace/data/RENI_HDR --latent-dim 100 --seed 42 \
    --invariant-function VN --equivariance SO2 --variant baseline \
    --training-paradigm standard --max-num-iterations "$steps" \
    --log-loss-variant "$variant" --skip-periodic-eval --direct-single-device \
    --experiment-name "$name" --timestamp fixed \
    --output-dir /workspace/outputs --vis tensorboard --quiet-local-writer \
    --keep-checkpoints --progress-jsonl "/workspace/outputs/$name/progress.jsonl"
