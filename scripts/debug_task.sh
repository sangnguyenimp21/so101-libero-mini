#!/usr/bin/env bash
# Inspect a three-cube task before any training (so101_libero/cli/debug_three_cubes.py; the server is headless, so
# every view is rendered to files whose paths are printed):
#   layouts     the five layouts (sort: x five starting permutations): front camera (the policy's 256x256 view), wrist
#               camera, overview; checks of settling, contacts, image margins, visibility, gaps, HOME clearance, expert
#   expert      one expert demonstration (MP4 + key frames) with the success detectors traced per step (CSV)
#   replay      one dataset episode replayed open-loop in the simulator next to its recorded frames
#   controller  the episode controller: stop on success, discarded actions, release, retreat, return HOME; and a
#               runaway policy that keeps acting, with and without stop-on-success
#   all         layouts + expert + controller (+ replay when the dataset exists)   [default]
#
#   bash scripts/debug_task.sh flip                          # everything, outputs in outputs/debug/flip/
#   bash scripts/debug_task.sh sort layouts                  # one mode
#   bash scripts/debug_task.sh pyramid expert controller --layout 3 --seed 7
#   bash scripts/debug_task.sh sort expert --layout 2 --perm YRG
#   bash scripts/debug_task.sh flip replay --episode 12      # dataset episode 12 of local/smolvla_flip_red
# Options: --layout K (0-4) --perm (sort) --seed S --trial T (controller) --episode E --repo-id ID --out DIR --viewer
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
source scripts/three_cubes_common.sh
three_cubes_task "${1:?usage: debug_task.sh flip|sort|pyramid|stack|green_right [layouts|expert|replay|controller|all ...] [options]}"
[ "$TASK_NAME" != multitask ] || { echo "debug one task at a time: flip, sort, pyramid, stack or green_right" >&2; exit 1; }
shift
MODES=()
while [ $# -gt 0 ] && [[ "$1" != -* ]]; do MODES+=("$1"); shift; done
[ ${#MODES[@]} -gt 0 ] || MODES=(all)
exec python -m so101_libero.cli.debug_three_cubes --task "$TASK_NAME" --repo-id "$REPO_ID" "${MODES[@]}" "$@"
