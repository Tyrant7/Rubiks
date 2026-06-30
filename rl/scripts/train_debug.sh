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

# Run identity
export RL_RUN_NAME="${RL_RUN_NAME:-resnet_sparse_rewards-$(date +%Y%m%d-%H%M%S)}"

# Episode structure
export RL_EPISODES="${RL_EPISODES:-20000}"
export RL_NUM_ENVS="${RL_NUM_ENVS:-256}"
export RL_LEARNING_STARTS="${RL_LEARNING_STARTS:-20000}"

# Evaluation
export RL_EVAL_EVERY="${RL_EVAL_EVERY:-1000}"
export RL_EVAL_EPISODES="${RL_EVAL_EPISODES:-100}"

# Logging & saving
export RL_LOG_EVERY="${RL_LOG_EVERY:-25}"
export RL_SAVE_EVERY="${RL_SAVE_EVERY:-1000}"

# Replay buffer
export RL_BUFFER_SIZE="${RL_BUFFER_SIZE:-500000}"
export RL_BATCH_SIZE="${RL_BATCH_SIZE:-512}"

# Optimizer
export RL_LEARNING_RATE="${RL_LEARNING_RATE:-3e-4}"
export RL_ALPHA_LR="${RL_ALPHA_LR:-3e-4}"
export RL_ADAM_EPS="${RL_ADAM_EPS:-1e-5}"

# TD learning
export RL_GAMMA="${RL_GAMMA:-0.99}"
export RL_TAU="${RL_TAU:-1}"

# Entropy / alpha
export RL_TARGET_ENTROPY_SCALE="${RL_TARGET_ENTROPY_SCALE:-0.2}"
export RL_LOG_ALPHA_INIT="${RL_LOG_ALPHA_INIT:--2.0}"

# Update schedule
export RL_UPDATE_EVERY="${RL_UPDATE_EVERY:-6}"
export RL_TARGET_NETWORK_FREQUENCY="${RL_TARGET_NETWORK_FREQUENCY:-2000}"

# Curriculum
export RL_CURRICULUM_THRESHOLD="${RL_CURRICULUM_THRESHOLD:-10}"
export RL_MAX_SCRAMBLE="${RL_MAX_SCRAMBLE:-11}"
export RL_CURRICULUM_MIN_EPISODES="${RL_CURRICULUM_MIN_EPISODES:-1}"
export RL_CLEAR_REPLAY_ON_ADVANCE="${RL_CLEAR_REPLAY_ON_ADVANCE:-false}"

cargo run --release -p rl
