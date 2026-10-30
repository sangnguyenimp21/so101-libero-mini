#!/usr/bin/env bash
# Collect exactly 100 expert demonstrations of the green_right task (green to the right of red) (see scripts/collect_task.sh for settings).
exec bash "$(dirname "${BASH_SOURCE[0]}")/collect_task.sh" green_right "$@"
