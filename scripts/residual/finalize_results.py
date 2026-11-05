"""Write the final result summary into NOTES.md after batch evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_START = "<!-- RESIDUAL_CURRENT_SUMMARY_START -->"
SUMMARY_END = "<!-- RESIDUAL_CURRENT_SUMMARY_END -->"
PHASE_START = "<!-- RESIDUAL_PHASE_E_START -->"
PHASE_END = "<!-- RESIDUAL_PHASE_E_END -->"


def fraction(summary: dict) -> str:
    return f"{summary['successes']}/{summary['episodes']} ({100 * summary['success_rate']:.1f}%)"


def make_current(report: dict) -> str:
    original = report["original"]
    target = report["target_sets"]
    shifted = report["all_shifted"]
    return "\n".join(
        [
            SUMMARY_START,
            "## Current summary",
            "",
            f"- Final residual model: `{report['residual_model']}`; SmolVLA stayed frozen.",
            f"- Target sets (axis 2 cm + diagonal 1.5 cm): SmolVLA {fraction(target['baseline'])}; "
            f"residual {fraction(target['residual'])}.",
            f"- Original 50 regression check: SmolVLA {fraction(original['baseline'])}; "
            f"residual {fraction(original['residual'])}.",
            f"- All 140 shifted states: SmolVLA {fraction(shifted['baseline'])}; residual {fraction(shifted['residual'])}.",
            "- Method limitation: the residual observes privileged simulator cube pose and is sim-only.",
            "- Final tables, plots, and improvement videos are under `docs/residual/results/final_evaluation/`.",
            SUMMARY_END,
        ]
    )


def make_phase(report: dict, evaluation_job: str) -> str:
    lines = [
        PHASE_START,
        "## Phase E / Step 10 — Final matched evaluation",
        "",
        f"- Dependent PBS evaluation job: `{evaluation_job}`.",
        f"- Evaluated model: `{report['residual_model']}` with frozen base `{report['base_policy']}`.",
        "- Protocol: exact same 50 original and 140 shifted initial states, seeds, horizon, EGL renderer, and success predicate as the saved baselines.",
        f"- Original: baseline **{fraction(report['original']['baseline'])}**, residual **{fraction(report['original']['residual'])}**.",
        f"- Target axis-2-cm + diagonal-1.5-cm sets: baseline **{fraction(report['target_sets']['baseline'])}**, "
        f"residual **{fraction(report['target_sets']['residual'])}**.",
        f"- All shifted: baseline **{fraction(report['all_shifted']['baseline'])}**, residual **{fraction(report['all_shifted']['residual'])}**.",
        "- Per-set results:",
    ]
    for set_id, values in report["per_set"].items():
        residual = values["residual"]
        motion = residual["motion_failures"]
        lines.append(
            f"  - `{set_id}`: baseline {fraction(values['baseline'])}; residual {fraction(residual)}; "
            f"residual failures stalled/moved/lifted = {motion['stalled']}/{motion['moved_but_missed']}/"
            f"{motion['lifted_but_not_flipped']}."
        )
    points = report.get("training_curve", [])
    if points:
        best = max(points, key=lambda row: row["success_rate"])
        lines.append(
            f"- Periodic frozen evaluation peaked at {best['successes']}/{best['episodes']} "
            f"({100 * best['success_rate']:.1f}%) at {best['timesteps']:,} transitions."
        )
    videos = report.get("improvement_videos", [])
    lines.append(f"- Residual-win videos: {len(videos)} ({', '.join(videos) if videos else 'none found'}).")
    lines += [
        "- Outputs: `docs/residual/results/final_evaluation/{results.json,episodes.csv,summary.md,success_vs_shift.png,training_curve.png,videos/}`.",
        "- Limitation: `privileged_cube_pose=true`; these residual results are simulation-only.",
        PHASE_END,
    ]
    return "\n".join(lines)


def replace_block(text: str, start: str, end: str, block: str, insert_after_title: bool) -> str:
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        return before.rstrip() + "\n\n" + block + after
    if insert_after_title:
        title, rest = text.split("\n", 1)
        return title + "\n\n" + block + "\n\n" + rest.lstrip()
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/residual/results/final_evaluation/results.json"))
    parser.add_argument("--notes", type=Path, default=Path("docs/residual/NOTES.md"))
    parser.add_argument("--evaluation-job", default="unknown")
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    if not report.get("complete") or len(report.get("episodes", [])) != 190:
        raise RuntimeError("refusing to finalize an incomplete final evaluation")
    text = args.notes.read_text()
    text = replace_block(text, SUMMARY_START, SUMMARY_END, make_current(report), insert_after_title=True)
    text = replace_block(text, PHASE_START, PHASE_END, make_phase(report, args.evaluation_job), insert_after_title=False)
    args.notes.write_text(text)


if __name__ == "__main__":
    main()
