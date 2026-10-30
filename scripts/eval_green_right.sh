#!/usr/bin/env bash
# Evaluate every saved checkpoint of the green_right run (settings: scripts/eval_task.sh).
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_task.sh" green_right "$@"
