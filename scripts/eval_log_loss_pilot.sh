#!/usr/bin/env bash
# ROOT GPU RUN_NAME [LATENT_STEPS] [OUTPUT_NAME]
set -euo pipefail
pilot_root="$1"
gpu="$2"
name="$3"
fit_steps="${4:-2500}"
output_name="${5:-validation_common}"
[[ "$name" =~ ^[a-z0-9_]+$ && "$output_name" =~ ^[a-z0-9_]+$ ]] || exit 2
case "$gpu" in 0|1|2) ;; *) exit 2 ;; esac
run_dir="$pilot_root/runs/$name"
exec >"$run_dir/$output_name.log" 2>&1
finish() { printf '%s\n' "$?" >"$run_dir/$output_name.exit_code"; }
trap finish EXIT
export CONTAINER_DIR=/home/userfs/j/jadg502/Personal_Staffstore/phd/.apptainer
export DATA_PATH=/home/userfs/j/jadg502/Personal_Staffstore/data
export MODEL_STORAGE_PATH=/home/userfs/j/jadg502/Personal_Staffstore/model-storage
export OUTPUTS_PATH="$pilot_root/runs"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu"
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_TORCH_HOME=/workspace/outputs/_cache/torch
export APPTAINERENV_MPLCONFIGDIR=/workspace/outputs/_cache/matplotlib
export APPTAINERENV_PYTHONUNBUFFERED=1
cd "$pilot_root/source"
bash .apptainer/apptainer.sh exec --ro -- env PYTHONPATH=/workspace/code:/workspace/code/scripts/figures:/overlay-packages \
    python /workspace/code/scripts/figures/eval_log_loss_pilot.py \
    --run "/workspace/outputs/$name/reni/fixed" --data /workspace/data/RENI_HDR \
    --output "/workspace/outputs/$name/$output_name" --latent-steps "$fit_steps" --seed 42
