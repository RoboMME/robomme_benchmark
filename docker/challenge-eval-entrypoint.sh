#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/test_videos
exec uv run --python 3.11 --no-sync python scripts/challenge_eval_policy.py "$@"
