#!/usr/bin/env bash
# Copy the evaluation summaries (results.json, a few KB each) into docs/results/eval/ so the README numbers are
# backed by versioned files (outputs/ is not versioned). Names: <run>__<checkpoint or variant>.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p docs/results/eval
for f in outputs/train/*/eval*/*/results.json outputs/eval/*/results.json outputs/eval/*/*/results.json; do
    [ -f "$f" ] || continue
    case "$f" in *verify*|*smoke*|*crashed_invalid*) continue ;; esac
    run=$(echo "$f" | sed -E 's#outputs/(train/)?##; s#/results.json##; s#/#__#g')
    cp "$f" "docs/results/eval/$run.json"
done
ls docs/results/eval | sed 's/^/  /'
