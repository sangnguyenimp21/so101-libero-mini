"""Final matched-state evaluation of SmolVLA alone versus PPO residual."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from so101_libero.cli.evaluate import classify, wilson
from so101_libero.cli.make_init_states import load_init_states
from so101_libero.residual.placements import EVALUATION_SETS
from so101_libero.residual.residual_env import ResidualConfig, ResidualSO101Env
from so101_libero.utils.video import label, write_mp4

DEFAULT_BASE_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_MODEL = Path("docs/residual/results/ppo_main/best_model.zip")
DEFAULT_TRAIN_DIR = Path("docs/residual/results/ppo_main")
DEFAULT_STATES = Path("docs/residual/placements")
DEFAULT_BASE_ORIGINAL = Path("docs/residual/results/baseline_original_50/results.json")
DEFAULT_BASE_SHIFTED = Path("docs/residual/results/baseline_shifted_140/results.json")
DEFAULT_STALLS = Path("docs/residual/results/baseline_shifted_140/stall_diagnostics/classification.csv")
DEFAULT_OUT = Path("docs/residual/results/final_evaluation")
FAILURE_TYPES = ("never_grasped", "dropped", "lifted_but_not_flipped", "timeout", "other")
MOTION_TYPES = ("stalled", "moved_but_missed", "lifted_but_not_flipped")


def broad_failure_type(success: bool, outcome: str, lifted: bool, steps: int, max_steps: int) -> str | None:
    if success:
        return None
    if outcome == "never_lifted":
        return "never_grasped"
    if outcome == "fell_off_table" or "fell_off" in outcome or outcome.startswith("dropped"):
        return "dropped"
    if outcome in {
        "still_held",
        "not_turned",
        "turned_but_not_flat",
        "turned_but_not_in_place_or_not_settled",
    } or lifted:
        return "lifted_but_not_flipped"
    if steps >= max_steps and not outcome.startswith("error"):
        return "timeout"
    return "other"


def summarize(rows: list[dict], prefix: str) -> dict:
    n = len(rows)
    successes = sum(bool(row[f"{prefix}_success"]) for row in rows)
    if n:
        lo, hi = wilson(successes, n)
        interval = [round(lo, 4), round(hi, 4)]
    else:
        interval = None
    failure_counts = Counter(
        row[f"{prefix}_failure_type"] for row in rows if not row[f"{prefix}_success"]
    )
    motion_counts = Counter(
        row.get(f"{prefix}_motion_category") for row in rows if not row[f"{prefix}_success"]
    )
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": round(successes / n, 4) if n else None,
        "wilson95": interval,
        "failure_types": {name: int(failure_counts[name]) for name in FAILURE_TYPES},
        "motion_failures": {name: int(motion_counts[name]) for name in MOTION_TYPES},
    }


def load_jobs(args) -> list[dict]:
    original_baseline = json.loads(args.baseline_original.read_text())["episodes"]
    shifted_baseline = json.loads(args.baseline_shifted.read_text())["episodes"]
    original_states = load_init_states("cube_flip")["states"]
    if len(original_states) != 50 or len(original_baseline) != 50:
        raise RuntimeError("expected exactly 50 original states and baseline rows")
    jobs = []
    for state, baseline in zip(original_states, original_baseline):
        jobs.append(
            {
                "set_id": "original_50",
                "state_index": int(state["index"]),
                "seed": int(state["seed"]),
                "init_state": state["init_state"],
                "offset_label": f"placement_{state['init_state'].get('placement_id', 'unknown')}",
                "offset_m": None,
                "training_region_tag": "original",
                "baseline": baseline,
            }
        )
    shifted_by_key = {
        (row["set_id"], int(row["state_index"])): row for row in shifted_baseline
    }
    for spec in EVALUATION_SETS:
        states = json.loads((args.states_dir / f"{spec.name}.json").read_text())["states"]
        for state in states:
            jobs.append(
                {
                    "set_id": spec.name,
                    "state_index": int(state["index"]),
                    "seed": int(state["seed"]),
                    "init_state": state["init_state"],
                    "offset_label": state["offset_label"],
                    "offset_m": state["offset_m"],
                    "training_region_tag": state["training_region_tag"],
                    "baseline": shifted_by_key[(spec.name, int(state["index"]))],
                }
            )
    if len(jobs) != 190:
        raise RuntimeError(f"expected 190 final evaluation states, found {len(jobs)}")
    return jobs


def load_baseline_motion(path: Path) -> dict[tuple[str, int], str]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[(row["set_id"], int(row["state_index"]))] = row["motion_category"]
    return result


def run_episode(model: PPO, env: ResidualSO101Env, job: dict, video_path: Path | None = None) -> dict:
    seed = job["seed"]
    torch.manual_seed(seed)
    obs, _ = env.reset(
        seed=seed,
        options={"init_state": job["init_state"], "offset_m": job["offset_m"] or (np.nan, np.nan)},
    )
    start = env.base_env.joint_state().copy()
    max_arm_change = 0.0
    total_reward = 0.0
    frames = []
    steps = 0
    while True:
        if video_path is not None:
            frames.append(label(env.render(), f"{job['set_id']} {job['offset_label']} t={steps}"))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        state = env.base_env.joint_state()
        max_arm_change = max(max_arm_change, float(np.max(np.abs(state[:5] - start[:5]))))
        if terminated or truncated:
            break
    success = bool(info["is_success"])
    lifted = bool(info["lifted"])
    failure = info.get("failure") or (None if success else "timeout")
    if success:
        outcome = "success"
    elif failure == "simulation diverged" or (failure and "fell off" in failure):
        outcome = failure.replace(" ", "_")
    else:
        outcome = classify(env.base_env, env.task, lifted)
    if success:
        motion_category = None
    elif lifted:
        motion_category = "lifted_but_not_flipped"
    elif max_arm_change < 5.0:
        motion_category = "stalled"
    else:
        motion_category = "moved_but_missed"
    if video_path is not None:
        frames.append(label(env.render(), f"{job['set_id']} {job['offset_label']} t={steps} success={success}"))
        write_mp4(frames, video_path, fps=env.base_env.cfg.control_hz)
    return {
        "success": success,
        "steps": steps,
        "outcome": outcome,
        "lifted": lifted,
        "failure_type": broad_failure_type(success, outcome, lifted, steps, env.task.max_steps),
        "motion_category": motion_category,
        "max_arm_joint_change_deg": round(max_arm_change, 5),
        "total_reward": round(total_reward, 6),
    }


def choose_improvement_videos(rows: list[dict], count: int) -> list[dict]:
    candidates = [row for row in rows if not row["baseline_success"] and row["residual_success"]]
    priorities = ("shift_2cm_axis", "shift_1p5cm_diagonal")
    candidates.sort(key=lambda row: (priorities.index(row["set_id"]) if row["set_id"] in priorities else 2,
                                     row["set_id"], row["state_index"]))
    selected = []
    while candidates and len(selected) < count:
        used = {(row["set_id"], row["offset_label"]) for row in selected}
        candidate = next(
            (row for row in candidates if (row["set_id"], row["offset_label"]) not in used),
            candidates[0],
        )
        selected.append(candidate)
        candidates.remove(candidate)
    return selected


def plot_success(rows: list[dict], out: Path) -> None:
    order = ["original_50", "shift_1p5cm_diagonal", "shift_2cm_axis",
             "shift_3cm_axis_feasible", "shift_4cm_axis_feasible"]
    labels = ["Original", "Diagonal 1.5 cm", "Axis 2 cm", "Axis 3 cm", "Axis 4 cm"]
    baseline = [summarize([r for r in rows if r["set_id"] == s], "baseline")["success_rate"] for s in order]
    residual = [summarize([r for r in rows if r["set_id"] == s], "residual")["success_rate"] for s in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, baseline, "o-", label="SmolVLA alone", linewidth=2)
    ax.plot(x, residual, "o-", label="SmolVLA + residual", linewidth=2)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_training_curve(train_dir: Path, out: Path) -> list[dict]:
    points = []
    initial_path = train_dir / "initial_evaluation.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text())
        points.append({"timesteps": 0, "success_rate": initial["success_rate"], "successes": initial["successes"],
                       "episodes": initial["episodes"]})
    eval_path = train_dir / "evaluations.jsonl"
    if eval_path.exists():
        for line in eval_path.read_text().splitlines():
            row = json.loads(line)
            points.append({k: row[k] for k in ("timesteps", "success_rate", "successes", "episodes")})
    points.sort(key=lambda row: row["timesteps"])
    if points:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p["timesteps"] for p in points], [p["success_rate"] for p in points], "o-", linewidth=2)
        ax.set_xlabel("Training transitions")
        ax.set_ylabel("Frozen target-set success rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
    return points


def write_summary(path: Path, report: dict) -> None:
    labels = {
        "original_50": "Original 50",
        "shift_2cm_axis": "Axis 2 cm",
        "shift_1p5cm_diagonal": "Diagonal 1.5 cm",
        "shift_3cm_axis_feasible": "Axis 3 cm feasible",
        "shift_4cm_axis_feasible": "Axis 4 cm feasible",
    }
    lines = ["# Final residual evaluation", "", "| set | SmolVLA | residual | delta | residual stalled / moved / lifted |",
             "|---|---:|---:|---:|---:|"]
    for set_id, values in report["per_set"].items():
        b, r = values["baseline"], values["residual"]
        m = r["motion_failures"]
        lines.append(
            f"| {labels[set_id]} | {b['successes']}/{b['episodes']} | {r['successes']}/{r['episodes']} | "
            f"{r['successes'] - b['successes']:+d} | {m['stalled']} / {m['moved_but_missed']} / "
            f"{m['lifted_but_not_flipped']} |"
        )
    lines += ["", "Success intervals and complete episode records are in `results.json`.", "",
              "![Success comparison](success_vs_shift.png)", "", "![Training curve](training_curve.png)", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, default=DEFAULT_BASE_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--baseline-original", type=Path, default=DEFAULT_BASE_ORIGINAL)
    parser.add_argument("--baseline-shifted", type=Path, default=DEFAULT_BASE_SHIFTED)
    parser.add_argument("--baseline-stalls", type=Path, default=DEFAULT_STALLS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--videos", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument("--limit", type=int, default=None, help="smoke test on first N matched states")
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        parser.error("run on a GPU node with MUJOCO_GL=egl")
    if not args.model.exists():
        parser.error(f"trained residual model does not exist: {args.model}")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    baseline_motion = load_baseline_motion(args.baseline_stalls)
    model = PPO.load(args.model, device="cpu")
    env = ResidualSO101Env(ResidualConfig(checkpoint=args.base_policy))
    rows = []
    started = time.time()
    try:
        for i, job in enumerate(jobs):
            result = run_episode(model, env, job)
            baseline = job["baseline"]
            baseline_lifted = bool(baseline["lifted"])
            baseline_failure = broad_failure_type(
                bool(baseline["success"]), baseline["outcome"], baseline_lifted,
                int(baseline["steps"]), env.task.max_steps,
            )
            if baseline["success"]:
                baseline_motion_category = None
            else:
                baseline_motion_category = baseline_motion.get((job["set_id"], job["state_index"]))
                if baseline_motion_category is None:
                    baseline_motion_category = "lifted_but_not_flipped" if baseline_lifted else "moved_but_missed"
            row = {
                "global_index": i,
                **{k: job[k] for k in ("set_id", "state_index", "seed", "offset_label", "offset_m", "training_region_tag")},
                "baseline_success": bool(baseline["success"]),
                "baseline_steps": int(baseline["steps"]),
                "baseline_outcome": baseline["outcome"],
                "baseline_lifted": baseline_lifted,
                "baseline_failure_type": baseline_failure,
                "baseline_motion_category": baseline_motion_category,
                **{f"residual_{k}": v for k, v in result.items()},
            }
            rows.append(row)
            print(
                f"{i + 1:03d}/{len(jobs)} {job['set_id']} {job['offset_label']} state={job['state_index']:02d}: "
                f"baseline={'success' if baseline['success'] else baseline['outcome']} residual={result['outcome']}",
                flush=True,
            )

        videos = []
        if args.limit is None and args.videos:
            selected = choose_improvement_videos(rows, args.videos)
            video_dir = args.out / "videos"
            video_dir.mkdir(exist_ok=True)
            job_lookup = {(j["set_id"], j["state_index"]): j for j in jobs}
            for row in selected:
                job = job_lookup[(row["set_id"], row["state_index"])]
                name = f"{row['set_id']}__{row['offset_label']}__state{row['state_index']:02d}_improvement.mp4"
                video_result = run_episode(model, env, job, video_dir / name)
                if video_result["success"] != row["residual_success"]:
                    raise RuntimeError(f"video rerun changed success for {name}")
                row["video"] = f"videos/{name}"
                videos.append(row["video"])
    finally:
        env.close()

    set_order = ["original_50"] + [spec.name for spec in EVALUATION_SETS]
    per_set = {
        set_id: {
            "baseline": summarize([r for r in rows if r["set_id"] == set_id], "baseline"),
            "residual": summarize([r for r in rows if r["set_id"] == set_id], "residual"),
        }
        for set_id in set_order if any(r["set_id"] == set_id for r in rows)
    }
    report = {
        "complete": args.limit is None and len(rows) == 190,
        "matched_state_protocol": True,
        "base_policy": str(args.base_policy),
        "residual_model": str(args.model),
        "privileged_cube_pose": True,
        "sim_only": True,
        "wall_time_s": round(time.time() - started, 1),
        "overall": {"baseline": summarize(rows, "baseline"), "residual": summarize(rows, "residual")},
        "original": {
            "baseline": summarize([r for r in rows if r["set_id"] == "original_50"], "baseline"),
            "residual": summarize([r for r in rows if r["set_id"] == "original_50"], "residual"),
        },
        "all_shifted": {
            "baseline": summarize([r for r in rows if r["set_id"] != "original_50"], "baseline"),
            "residual": summarize([r for r in rows if r["set_id"] != "original_50"], "residual"),
        },
        "target_sets": {
            "baseline": summarize([r for r in rows if r["set_id"] in ("shift_2cm_axis", "shift_1p5cm_diagonal")], "baseline"),
            "residual": summarize([r for r in rows if r["set_id"] in ("shift_2cm_axis", "shift_1p5cm_diagonal")], "residual"),
        },
        "per_set": per_set,
        "per_training_region": {
            tag: {
                "baseline": summarize([r for r in rows if r["training_region_tag"] == tag], "baseline"),
                "residual": summarize([r for r in rows if r["training_region_tag"] == tag], "residual"),
            }
            for tag in ("in-training-region", "out-of-training-region")
            if any(r["training_region_tag"] == tag for r in rows)
        },
        "improvement_videos": videos if args.limit is None else [],
        "episodes": rows,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [k for k in rows[0] if k != "offset_m"] + ["video"]
    with (args.out / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    if args.limit is None:
        plot_success(rows, args.out / "success_vs_shift.png")
        points = plot_training_curve(args.train_dir, args.out / "training_curve.png")
        report["training_curve"] = points
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        write_summary(args.out / "summary.md", report)


if __name__ == "__main__":
    main()
