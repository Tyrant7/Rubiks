#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

default_libtorch="$HOME/Downloads/libtorch-shared-with-deps-2.10.0+cu130/libtorch"
using_default_libtorch=0
if [[ -z "${LIBTORCH:-}" && -d "$default_libtorch" ]]; then
    export LIBTORCH="$default_libtorch"
    using_default_libtorch=1
fi

if [[ -n "${LIBTORCH:-}" ]]; then
    if [[ "$using_default_libtorch" == "1" ]]; then
        export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
    fi
    export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
else
    echo "LIBTORCH is unset; tch will use its download-libtorch configuration." >&2
fi

export RL_RUN_NAME="${RL_RUN_NAME:-debug-$(date +%Y%m%d-%H%M%S)}"
export RL_EPISODES="${RL_EPISODES:-20000}"
export RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"
export RL_BUFFER_SIZE="${RL_BUFFER_SIZE:-200000}"
export RL_CURRICULUM_THRESHOLD="${RL_CURRICULUM_THRESHOLD:-10}"
export RL_CURRICULUM_MIN_EPISODES="${RL_CURRICULUM_MIN_EPISODES:-1}"
export RL_TARGET_ENTROPY_SCALE="${RL_TARGET_ENTROPY_SCALE:-0.2}"
export RL_LOG_ALPHA_INIT="${RL_LOG_ALPHA_INIT:--2.0}"
export RL_ALPHA_LR="${RL_ALPHA_LR:-3e-4}"
export RL_ADAM_EPS="${RL_ADAM_EPS:-1e-4}"
export RL_TAU="${RL_TAU:-0.000125}"
export RL_CLEAR_REPLAY_ON_ADVANCE="${RL_CLEAR_REPLAY_ON_ADVANCE:-false}"
export RL_UPDATE_EVERY="${RL_UPDATE_EVERY:-4}"
export RL_TARGET_NETWORK_FREQUENCY="${RL_TARGET_NETWORK_FREQUENCY:-1}"
export RL_LEARNING_STARTS="${RL_LEARNING_STARTS:-5000}"
export RL_NUM_ENVS="${RL_NUM_ENVS:-16}"
export RL_EVAL_EVERY="${RL_EVAL_EVERY:-0}"
export RL_EVAL_EPISODES="${RL_EVAL_EPISODES:-64}"
export RL_LOG_EVERY="${RL_LOG_EVERY:-25}"
export RL_SAVE_EVERY="${RL_SAVE_EVERY:-1000}"

cargo run --release -p rl
