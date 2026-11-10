"""Write the final result summary into NOTES.md after batch evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_START = "<!-- RESIDUAL_CURRENT_SUMMARY_START -->"
SUMMARY_END = "<!-- RESIDUAL_CURRENT_SUMMARY_END -->"


def fraction(summary: dict) -> str:
    return f"{summary['successes']}/{summary['episodes']} ({100 * summary['success_rate']:.1f}%)"


def make_current(reports: dict[str, dict]) -> str:
    lines = [SUMMARY_START, "## Current summary", "", "- SmolVLA stayed frozen in every run."]
    for label, report in sorted(reports.items()):
        target, original = report["target_sets"], report["original"]
        paired = report["paired_target_sets"]
        lines.append(
            f"- **{label} best-eval:** target {fraction(target['residual'])} vs baseline "
            f"{fraction(target['baseline'])}; fixed/broken={paired['best']['fixed']}/{paired['best']['broken']} "
            f"(McNemar p={paired['best']['mcnemar_exact_two_sided_p']}); original {fraction(original['residual'])}."
        )
        lines.append(
            f"- **{label} final:** target {fraction(target['final'])}; "
            f"fixed/broken={paired['final']['fixed']}/{paired['final']['broken']} "
            f"(McNemar p={paired['final']['mcnemar_exact_two_sided_p']})."
        )
    lines += [
        "- Method limitation: the residual observes privileged simulator cube pose and is sim-only.",
        "- Variant tables, plots, and improvement videos are under `docs/residual/results/final_evaluation_*/`.",
        SUMMARY_END,
    ]
    return "\n".join(lines)


def make_phase(report: dict, evaluation_job: str, label: str, output_dir: Path) -> str:
    phase_start = f"<!-- RESIDUAL_PHASE_E_{label.upper()}_START -->"
    phase_end = f"<!-- RESIDUAL_PHASE_E_{label.upper()}_END -->"
    paired = report["paired_target_sets"]
    lines = [
        phase_start,
        f"## Phase E / Step 10 — {label} matched evaluation",
        "",
        f"- Dependent PBS evaluation job: `{evaluation_job}`.",
        f"- Best-eval model: `{report['residual_model']}`; final model: `{report['final_model']}`; frozen base: `{report['base_policy']}`.",
        "- Protocol: exact same 50 original and 140 shifted initial states, seeds, horizon, EGL renderer, and success predicate as the saved baselines.",
        f"- Original: baseline **{fraction(report['original']['baseline'])}**, best **{fraction(report['original']['residual'])}**, "
        f"final **{fraction(report['original']['final'])}**.",
        f"- Target axis-2-cm + diagonal-1.5-cm sets: baseline **{fraction(report['target_sets']['baseline'])}**, "
        f"best **{fraction(report['target_sets']['residual'])}**, final **{fraction(report['target_sets']['final'])}**.",
        f"- Target paired best: fixed={paired['best']['fixed']}, broken={paired['best']['broken']}, "
        f"McNemar exact p={paired['best']['mcnemar_exact_two_sided_p']}; final: fixed={paired['final']['fixed']}, "
        f"broken={paired['final']['broken']}, p={paired['final']['mcnemar_exact_two_sided_p']}.",
        f"- All shifted: baseline **{fraction(report['all_shifted']['baseline'])}**, best **{fraction(report['all_shifted']['residual'])}**, "
        f"final **{fraction(report['all_shifted']['final'])}**.",
        "- Per-set results:",
    ]
    for set_id, values in report["per_set"].items():
        residual = values["residual"]
        motion = residual["motion_failures"]
        lines.append(
            f"  - `{set_id}`: baseline {fraction(values['baseline'])}; best {fraction(residual)}; "
            f"final {fraction(values['final'])}; "
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
        f"- Outputs: `{output_dir}/{{results.json,episodes.csv,summary.md,success_vs_shift.png,training_curve.png,videos/}}`.",
        "- Limitation: `privileged_cube_pose=true`; these residual results are simulation-only.",
        phase_end,
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
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    if not report.get("complete") or len(report.get("episodes", [])) != 190:
        raise RuntimeError("refusing to finalize an incomplete final evaluation")
    result_root = args.results.parent.parent
    reports = {}
    for candidate in sorted(result_root.glob("final_evaluation_*/results.json")):
        candidate_report = json.loads(candidate.read_text())
        if candidate_report.get("complete") and len(candidate_report.get("episodes", [])) == 190:
            reports[candidate.parent.name.removeprefix("final_evaluation_")] = candidate_report
    reports[args.label] = report
    text = args.notes.read_text()
    text = replace_block(text, SUMMARY_START, SUMMARY_END, make_current(reports), insert_after_title=True)
    phase_start = f"<!-- RESIDUAL_PHASE_E_{args.label.upper()}_START -->"
    phase_end = f"<!-- RESIDUAL_PHASE_E_{args.label.upper()}_END -->"
    text = replace_block(
        text,
        phase_start,
        phase_end,
        make_phase(report, args.evaluation_job, args.label, args.results.parent),
        insert_after_title=False,
    )
    args.notes.write_text(text)


if __name__ == "__main__":
    main()
