#!/usr/bin/env bash
# Simulated inference with the full episode logic: HOME -> SmolVLA -> success detected (simulator ground truth) ->
# policy stopped, queued actions discarded -> safe release -> safe retreat -> HOME.
#
#   bash scripts/run_sim.sh --task flip --checkpoint outputs/flip_red/checkpoints/060000/pretrained_model
#   bash scripts/run_sim.sh --task sort --checkpoint <ckpt> --episodes 10 --layout 2 3 --video 10
#   bash scripts/run_sim.sh --task pyramid --policy expert           # the controller with the scripted expert
# Defaults added unless given: --episodes 5 --video 5 --out outputs/run_sim/<task>_<time>. Every option of
# python -m so101_libero.cli.run_episodes works (--detector behavior shows when the real-robot rule would stop).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
ARGS=("$@")
has() { local a; for a in "${ARGS[@]}"; do [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0; done; return 1; }
TASK=sim
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in --task) TASK="${ARGS[$((i + 1))]:-sim}" ;; --task=*) TASK="${ARGS[$i]#--task=}" ;; esac
done
has --episodes || ARGS+=(--episodes 5)
has --video || ARGS+=(--video 5)
has --out || ARGS+=(--out "outputs/run_sim/${TASK}_$(date +%Y%m%d_%H%M%S)")
exec python -m so101_libero.cli.run_episodes --robot sim "${ARGS[@]}"
