#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

port="${TENSORBOARD_PORT:-6009}"
tensorboard --logdir "$PWD/rl/logs" --host 127.0.0.1 --port "$port"
