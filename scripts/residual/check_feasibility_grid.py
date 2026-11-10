"""Map scripted-expert feasibility over cube-group x/y translations.

This is a diagnostic screen, not a frozen evaluation-set generator.  It runs
one full physics-only expert episode at every 0.5 cm grid point from -4 cm to
+4 cm on each axis.  Every cell has a unique, recorded seed.  Failed cells are
expected and retained; they define the feasibility boundary.

Run on a GPU node from the repository root:

    MUJOCO_GL=egl python -u scripts/residual/check_feasibility_grid.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from so101_libero.cli.generate import episode_limit
from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import run_expert_episode
from so101_libero.residual.placements import shifted_init_state
from so101_libero.tasks.registry import get_task

GRID_HALF_CM = tuple(range(-8, 9))  # -4.0 cm through +4.0 cm in 0.5 cm increments
SEED_BASE = 32_000_000
FIELDS = (
    "index",
    "seed",
    "dx_cm",
    "dy_cm",
    "cube_x_m",
    "cube_y_m",
    "cube_radius_from_pan_cm",
    "success",
    "failure",
    "steps",
    "success_step",
    "wall_time_s",
)


def require_gpu_node() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("the feasibility grid must run on a GPU node") from exc
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise RuntimeError("the feasibility grid must run on a GPU node, but no NVIDIA GPU is visible")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl on the GPU node")
    print(f"GPU node confirmed: {', '.join(names)}; MUJOCO_GL=egl", flush=True)


def grid_cells() -> list[dict]:
    cells = []
    for iy, hy in enumerate(GRID_HALF_CM):
        for ix, hx in enumerate(GRID_HALF_CM):
            index = iy * len(GRID_HALF_CM) + ix
            cells.append(
                {
                    "index": index,
                    "seed": SEED_BASE + index,
                    "dx_cm": hx / 2,
                    "dy_cm": hy / 2,
                    "dx_m": hx / 200,
                    "dy_m": hy / 200,
                }
            )
    return cells


def read_rows(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["index"]): row for row in rows}


def write_rows(path: Path, rows: dict[int, dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index in sorted(rows):
            writer.writerow({key: rows[index].get(key, "") for key in FIELDS})


def bool_value(value) -> bool:
    return value is True or str(value).lower() == "true"


def render_map(rows: dict[int, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n = len(GRID_HALF_CM)
    feasible = np.zeros((n, n), dtype=np.uint8)
    for row in rows.values():
        ix = int(round((float(row["dx_cm"]) + 4.0) * 2))
        iy = int(round((float(row["dy_cm"]) + 4.0) * 2))
        feasible[iy, ix] = 1 if bool_value(row["success"]) else 0

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    ax.imshow(
        feasible,
        origin="lower",
        interpolation="nearest",
        extent=(-4.25, 4.25, -4.25, 4.25),
        cmap=ListedColormap(["#d9534f", "#45a049"]),
        vmin=0,
        vmax=1,
    )
    ticks = np.arange(-4, 4.1, 1.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks(np.arange(-4.0, 4.1, 0.5), minor=True)
    ax.set_yticks(np.arange(-4.0, 4.1, 0.5), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Group x offset (cm): backward  ←  0  →  forward")
    ax.set_ylabel("Group y offset (cm): right  ←  0  →  left")
    ax.set_title("Cube-flip scripted-expert feasibility (one unique seed per cell)")
    ax.legend(
        handles=[Patch(facecolor="#45a049", label="expert success"), Patch(facecolor="#d9534f", label="expert failure")],
        loc="upper right",
        framealpha=0.95,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/residual/placements"))
    parser.add_argument("--restart", action="store_true", help="ignore and overwrite an existing partial table")
    args = parser.parse_args()

    require_gpu_node()
    args.out.mkdir(parents=True, exist_ok=True)
    table_path = args.out / "expert_feasibility_grid.csv"
    map_path = args.out / "expert_feasibility_grid.png"
    summary_path = args.out / "expert_feasibility_grid_summary.json"
    rows = {} if args.restart else read_rows(table_path)

    task = get_task("cube_flip")
    env = SO101TabletopEnv(
        task,
        EnvConfig(render=False, terminate_on_success=False, max_steps=episode_limit(task.task_id)),
    )
    env.reset(options={"init_state": shifted_init_state(0.0, 0.0)})
    pan_xy = env.data.xpos[env.model.body("shoulder").id][:2].copy()
    cells = grid_cells()
    try:
        for cell in cells:
            if cell["index"] in rows:
                continue
            state = shifted_init_state(cell["dx_m"], cell["dy_m"])
            cube_x, cube_y, _ = state["items"]["red_cube"]
            t0 = time.perf_counter()
            ep = run_expert_episode(env, cell["seed"], init_state=state)
            row = {
                "index": cell["index"],
                "seed": cell["seed"],
                "dx_cm": cell["dx_cm"],
                "dy_cm": cell["dy_cm"],
                "cube_x_m": round(cube_x, 6),
                "cube_y_m": round(cube_y, 6),
                "cube_radius_from_pan_cm": round(100 * float(np.linalg.norm(np.array([cube_x, cube_y]) - pan_xy)), 4),
                "success": ep.success,
                "failure": ep.failure or "",
                "steps": ep.steps,
                "success_step": "" if ep.success_step is None else ep.success_step,
                "wall_time_s": round(time.perf_counter() - t0, 3),
            }
            rows[cell["index"]] = row
            write_rows(table_path, rows)  # resumable after every cell
            status = "PASS" if ep.success else f"FAIL ({ep.failure})"
            print(
                f"cell {cell['index'] + 1:03d}/{len(cells)} "
                f"dx={cell['dx_cm']:+.1f} cm dy={cell['dy_cm']:+.1f} cm "
                f"seed={cell['seed']} -> {status}",
                flush=True,
            )
    finally:
        env.close()

    if len(rows) != len(cells):
        raise RuntimeError(f"grid incomplete: {len(rows)}/{len(cells)} cells in {table_path}")
    render_map(rows, map_path)
    successes = sum(bool_value(row["success"]) for row in rows.values())
    failures: dict[str, int] = {}
    for row in rows.values():
        if not bool_value(row["success"]):
            failure = str(row["failure"])
            failures[failure] = failures.get(failure, 0) + 1
    summary = {
        "task_id": task.task_id,
        "grid_cm": {"min": -4.0, "max": 4.0, "step": 0.5, "cells": len(cells)},
        "seed_base": SEED_BASE,
        "one_unique_seed_per_cell": True,
        "successes": successes,
        "failures": len(cells) - successes,
        "failure_reasons": failures,
        "table": str(table_path),
        "map": str(map_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"complete: {successes}/{len(cells)} feasible -> {table_path}, {map_path}", flush=True)


if __name__ == "__main__":
    main()
