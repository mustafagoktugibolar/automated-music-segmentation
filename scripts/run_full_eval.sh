#!/usr/bin/env bash
# Run the full-dataset segmentation evaluation inside the worker container.
# scripts/ is not volume-mounted, so the eval scripts are copied in first.
#
# Usage:
#   scripts/run_full_eval.sh [--label my-experiment] [--params '{"novelty_prominence":0.1}']
#                            [--max-tracks 50] [--concurrency 3]
#
# Results land in data/eval_runs/<run_id>/ (mounted on the host) and one
# summary line is appended to data/eval_runs/history.csv per run.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${CONTAINER:-worker-user-code}"

docker exec "$CONTAINER" mkdir -p /app/scripts
docker cp "$DIR/batch_eval.py" "$CONTAINER":/app/scripts/
docker cp "$DIR/full_dataset_eval.py" "$CONTAINER":/app/scripts/
docker exec -e PYTHONPATH=/app "$CONTAINER" python /app/scripts/full_dataset_eval.py "$@"
