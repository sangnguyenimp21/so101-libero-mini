#!/usr/bin/env bash
# Evaluate one checkpoint under several inference settings (same fixed init states), then print a summary table.
#
#   bash scripts/eval_variants.sh CHECKPOINT_DIR OUT_DIR "TASKS" [VARIANTS...]
#   bash scripts/eval_variants.sh outputs/train/act_t1_200/checkpoints/last/pretrained_model outputs/eval/act_t1_200 \
#        cylinder_to_tray n50 n25 n10 te
#
# Variants: nK = execute K actions per inference (ACT/Diffusion/SmolVLA n_action_steps); te = ACT temporal
# ensembling (coefficient 0.01, re-plans every step); default = the checkpoint's own setting.
# Env: EPISODES [50] VIDEOS [4] WORKERS [6] DEVICE [auto]
set -euo pipefail
CKPT="${1:?checkpoint dir}"; OUT="${2:?output dir}"; TASKS="${3:?tasks}"; shift 3
VARIANTS=("${@:-default}")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS=()
for v in "${VARIANTS[@]}"; do
    case "$v" in
        default) ARGS=() ;;
        te) ARGS=(--temporal-ensemble 0.01) ;;
        n*) ARGS=(--n-action-steps "${v#n}") ;;
        *) echo "unknown variant $v" >&2; exit 1 ;;
    esac
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/$v/results.json')).get('complete', True) else 1)" \
            2>/dev/null; then  # missing or incomplete: (re)run
        # shellcheck disable=SC2086
        bash "$ROOT/scripts/env.sh" python -m so101_libero.cli.evaluate --policy "$CKPT" --tasks $TASKS \
            --episodes "${EPISODES:-50}" --videos "${VIDEOS:-4}" --workers "${WORKERS:-6}" \
            ${DEVICE:+--device "$DEVICE"} --out "$OUT/$v" "${ARGS[@]}" | grep -E '"overall"|success_rate' | head -2
    fi
    RESULTS+=("$OUT/$v/results.json")
done
bash "$ROOT/scripts/env.sh" python -m so101_libero.cli.summarize "${RESULTS[@]}"
