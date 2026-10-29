#!/usr/bin/env bash
# Sanity check of a three-cube dataset: episodes, task labels, camera keys, state/action dimensions, episode lengths,
# NaNs, missing frames, layout distribution (sort: starting permutations too). See so101_libero/cli/check_dataset.py.
# The report is also written to results/dataset_check_<dataset name>.json.
#
#   bash scripts/check_dataset.sh flip | sort | pyramid | multitask [--decode-all]
#   REPO_ID=local/my_dataset bash scripts/check_dataset.sh flip       # another dataset
# The deeper validation (LeRobot loader + open-loop replay of recorded actions in the simulator) runs in the collect
# scripts; on its own: python -m so101_libero.cli.validate_dataset --repo-id <repo id> --replay-episodes 10
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
source scripts/three_cubes_common.sh
_repo="${REPO_ID:-}"
three_cubes_task "${1:?usage: check_dataset.sh flip|sort|pyramid|multitask [--decode-all]}"
REPO_ID="${_repo:-$REPO_ID}"
N=$([ "$TASK_NAME" = multitask ] && echo 300 || echo 100)
mkdir -p "$RESULTS_DIR"
exec python -m so101_libero.cli.check_dataset --repo-id "$REPO_ID" --expect-episodes "${EXPECT_EPISODES:-$N}" \
    --json "$RESULTS_DIR/dataset_check_$(basename "$REPO_ID").json" "${@:2}"
