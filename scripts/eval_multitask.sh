#!/usr/bin/env bash
# Evaluate every saved checkpoint of the multitask run (settings: scripts/eval_task.sh).
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_task.sh" multitask "$@"
