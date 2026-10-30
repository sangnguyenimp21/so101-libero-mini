#!/usr/bin/env bash
# Collect exactly 100 expert demonstrations of the stack task (yellow on red) (see scripts/collect_task.sh for settings).
exec bash "$(dirname "${BASH_SOURCE[0]}")/collect_task.sh" stack "$@"
