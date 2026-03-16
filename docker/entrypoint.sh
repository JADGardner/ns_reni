#!/bin/bash
# Entrypoint for the ns_reni research container.
# Installs mounted code editably (fast, no deps) then runs the user command.
set -e

# Ensure HOME exists and is writable
export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME" 2>/dev/null || true

# Copy wandb credentials if available
if [ -f /tmp/.netrc ]; then
    cp -f /tmp/.netrc "$HOME/.netrc" 2>/dev/null || true
fi

# Activate conda (disable -e temporarily; nerfstudio completions have syntax issues)
set +e
eval "$(conda shell.bash hook)"
conda activate research 2>/dev/null
set -e

# Install mounted ns_reni code editably (--no-deps: all deps are in the image)
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    rm -rf "$PROJECT_ROOT"/*.egg-info 2>/dev/null || true
    pip install -e "$PROJECT_ROOT" --no-deps --quiet 2>/dev/null || true
fi

# Run whatever command was passed
exec "$@"
