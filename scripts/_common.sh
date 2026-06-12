#!/bin/bash
# Generic boilerplate for training/eval scripts.
#
# Source this after setting SCRIPT_DIR and defining:
#   _build_train_cmd()  — populate _CMD array and _CUDA for training
#   _build_eval_cmd()   — populate _CMD array and _CUDA for evaluation
#
# Optional helpers you can call inside your builders:
#   _wrap_torchrun      — auto-wrap _CMD with torchrun if GPU_IDS has multiple GPUs

# ── Resolve paths ────────────────────────────────────────────

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env from parent of project dir (survives git resets)
_dotenv="$(dirname "$PROJECT_ROOT")/.env"
[ -f "$_dotenv" ] && { set -a; source "$_dotenv"; set +a; }

: "${CONDA_PATH:?CONDA_PATH not set — check .env}"
: "${CONDA_ENV:?CONDA_ENV not set — check .env}"

set -euo pipefail

# ── Conda (only activate if not already in the target env) ───

if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
    set +e
    __conda_setup="$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
    if [ $? -eq 0 ] && [ -n "$__conda_setup" ]; then eval "$__conda_setup"
    elif [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then . "${CONDA_PATH}/etc/profile.d/conda.sh"
    else export PATH="${CONDA_PATH}/bin:$PATH"
    fi
    unset __conda_setup
    conda activate "${CONDA_ENV}"
    set -e
fi

# ── Command helpers ──────────────────────────────────────────

# Print _CMD array as a safely-quoted shell command for tmux send-keys.
_print_cmd() {
    printf 'CUDA_VISIBLE_DEVICES=%s' "${_CUDA}"
    local elem
    for elem in "${_CMD[@]}"; do
        printf " '%s'" "${elem//\'/\'\\\'\'}"
    done
    printf '\n'
}

# Auto-wrap _CMD with torchrun if GPU_IDS contains multiple GPUs.
_wrap_torchrun() {
    local gpu_ids="${GPU_IDS:-0}"
    local num_gpus
    num_gpus=$(echo "$gpu_ids" | tr ',' '\n' | wc -l | tr -d ' ')

    if [ "$num_gpus" -gt 1 ]; then
        local master_port=29500
        while ss -tlnp 2>/dev/null | grep -q ":${master_port} " && [ "$master_port" -lt 29600 ]; do
            master_port=$((master_port + 1))
        done
        _CMD=(torchrun --nproc_per_node="${num_gpus}" --master_port="${master_port}" "${_CMD[@]}")
    fi

    export _CUDA="${gpu_ids}"
}

# ── tmux: launch in persistent session via send-keys ─────────

SESSION="${SESSION:-${MODE:-train}_${MODEL:-default}}"

_launch_in_tmux() {
    [ "${MODE:-train}" = "eval" ] && _build_eval_cmd || _build_train_cmd

    tmux new-session -d -s "${SESSION}" -c "${PROJECT_ROOT}"
    tmux send-keys -t "${SESSION}" \
        "source '${CONDA_PATH}/etc/profile.d/conda.sh' && conda activate '${CONDA_ENV}'" C-m
    tmux send-keys -t "${SESSION}" "cd '${PROJECT_ROOT}'" C-m
    tmux send-keys -t "${SESSION}" "$(_print_cmd); echo; echo '[done] press Ctrl+D to exit'; exec bash" C-m

    echo "Launched in tmux session '${SESSION}'. Attaching..."
    exec tmux attach -t "${SESSION}"
}

if [ -n "${NO_TMUX:-}" ]; then
    :
elif [ -n "${TMUX:-}" ]; then
    :
elif ! command -v tmux &>/dev/null; then
    :
elif tmux has-session -t "${SESSION}" 2>/dev/null; then
    _alive=$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' 2>/dev/null \
        | xargs -I{} ps --ppid {} -o comm= 2>/dev/null || true)
    if echo "$_alive" | grep -qE 'python|torchrun'; then
        echo "Session '${SESSION}' is running — attaching."
        exec tmux attach -t "${SESSION}"
    else
        echo "Session '${SESSION}' exists but idle — restarting."
        tmux kill-session -t "${SESSION}"
        _launch_in_tmux
    fi
else
    _launch_in_tmux
fi

# ── Inline execution (no tmux, or already inside tmux) ──────

if [ "${MODE:-train}" = "eval" ]; then
    _build_eval_cmd
else
    _build_train_cmd
fi

CUDA_VISIBLE_DEVICES="${_CUDA}" "${_CMD[@]}"
