"""Summarize evaluation results (results.json files) as a Markdown table.

    bash scripts/env.sh python -m so101_libero.cli.summarize outputs/eval/*/results.json outputs/train/*/eval*/*/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()
    rows = []
    for p in args.results:
        r = json.loads(p.read_text())
        pol = r["policy"]
        name = str(p.parent) if p.name == "results.json" else p.stem  # docs/results/eval/<run>.json copies
        extra = ", ".join(f"{k}={v}" for k, v in pol.items() if k in ("n_action_steps", "temporal_ensemble_coeff"))
        for task, t in r["per_task"].items():
            lo, hi = t["wilson95"]
            outcomes = ", ".join(f"{k} {v}" for k, v in sorted(t["outcomes"].items(), key=lambda kv: -kv[1]))
            rows.append(f"| {name} | {pol.get('type', 'expert')}{' (' + extra + ')' if extra else ''} | {task} | "
                        f"{t['successes']}/{t['episodes']} = {100 * t['success_rate']:.0f}% | "
                        f"[{100 * lo:.0f}, {100 * hi:.0f}] | {r.get('render_backend')} | {outcomes} |")
    print("| run | policy | task | success | Wilson 95% CI | renderer | outcomes |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
