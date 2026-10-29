#!/usr/bin/env bash
# Collect exactly 100 expert demonstrations of the sort task (see scripts/collect_task.sh for settings).
exec bash "$(dirname "${BASH_SOURCE[0]}")/collect_task.sh" sort "$@"
