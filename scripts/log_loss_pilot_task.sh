#!/usr/bin/env bash
# Completion-triggered train -> common validation refit, with no polling loop.
set -euo pipefail
pilot_root="$1"
gpu="$2"
variant="$3"
name="$4"
[[ "$name" =~ ^[a-z0-9_]+$ ]] || exit 2
mkdir -p "$pilot_root/tasks"
[[ ! -e "$pilot_root/tasks/$name.pid" ]] || exit 2
printf '%s\n' "$$" >"$pilot_root/tasks/$name.pid"
finish() { printf '%s\n' "$?" >"$pilot_root/tasks/$name.exit_code"; }
trap finish EXIT
bash "$pilot_root/source/scripts/run_log_loss_pilot.sh" "$pilot_root" "$gpu" "$variant" "$name" 50001
bash "$pilot_root/source/scripts/eval_log_loss_pilot.sh" "$pilot_root" "$gpu" "$name" 2500 validation_common
