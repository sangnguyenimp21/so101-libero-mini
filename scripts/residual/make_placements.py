"""Create, validate, and render shifted cube-flip evaluation states.

This command is intentionally strict: it stops on the first unsafe layout or
scripted-expert failure.  It never substitutes another seed, so the frozen
evaluation set cannot be silently selected to make the expert look better.

Run on a GPU node from the repository root:

    MUJOCO_GL=egl python -u scripts/residual/make_placements.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from PIL import Image

from so101_libero.cli.generate import episode_limit
from so101_libero.cli.make_init_states import rules_hash, scene_hash
from so101_libero.cli.render_placements import box_corners, object_corners, project
from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import run_expert_episode
from so101_libero.residual.placements import (
    EVALUATION_SETS,
    TRAIN_X_RANGE_M,
    TRAIN_Y_RANGE_M,
    evaluation_entries,
    shifted_init_state,
)
from so101_libero.tasks.base import _gap_to_tray
from so101_libero.tasks.registry import get_task
from so101_libero.utils.video import label


def require_gpu_node() -> None:
    """Refuse to perform validation/rendering without a visible NVIDIA GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("placement validation must run on a GPU node with nvidia-smi available") from exc
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise RuntimeError("placement validation must run on a GPU node, but no NVIDIA GPU is visible")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl so placement rendering uses the GPU")
    print(f"GPU node confirmed: {', '.join(names)}; MUJOCO_GL=egl", flush=True)


def contacts_for(env: SO101TabletopEnv, name: str) -> list[str]:
    m, d, geoms = env.model, env.data, env.h.item_geoms[name]
    return sorted(
        {
            m.geom(contact.geom2 if contact.geom1 in geoms else contact.geom1).name
            for contact in d.contact[: d.ncon]
            if contact.geom1 in geoms or contact.geom2 in geoms
        }
    )


def geometry_report(env: SO101TabletopEnv, state: dict, margin_px: float = 3.0) -> dict:
    """Check settling, contacts, spacing, table bounds, and front-camera visibility."""
    env.reset(options={"init_state": state})
    m, d, h, ws = env.model, env.data, env.h, env.world
    seg = mujoco.Renderer(m, 256, 256)
    seg.enable_segmentation_rendering()
    report: dict = {"objects": {}, "object_gaps_cm": {}, "tray_gaps_cm": {}, "ok": True}

    names = list(h.objects)
    for name in names:
        x, y, yaw = state["items"][name]
        pos = ws.pos(name)
        settle_shift = float(np.linalg.norm(pos[:2] - (x, y)))
        yaw_change = abs((ws.yaw(name) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if h.objects[name].yaw_symmetry:
            yaw_change = min(yaw_change, abs(h.objects[name].yaw_symmetry - yaw_change))
        contacts = contacts_for(env, name)
        corners = object_corners(env, name)
        uv = project(m, d, "front", corners, 256)
        in_front = bool(np.all(uv >= margin_px) and np.all(uv <= 256 - margin_px))
        seg.update_scene(d, camera="front")
        ids = seg.render()[..., 0]
        visible_pixels = int(np.isin(ids, list(h.item_geoms[name])).sum())
        radius = h.objects[name].footprint_radius
        in_table = bool(
            -0.22 + radius <= pos[0] <= 0.62 - radius
            and -0.50 + radius <= pos[1] <= 0.50 - radius
        )
        item = {
            "settle_shift_mm": round(settle_shift * 1e3, 3),
            "yaw_change_deg": round(math.degrees(yaw_change), 3),
            "height_mm": round(float(pos[2]) * 1e3, 3),
            "contacts": contacts,
            "inside_table": in_table,
            # Informational: shifted objects may cross the conservative three-pixel
            # image margin.  Actual visibility is enforced with visible_pixels.
            "fully_inside_front_image": in_front,
            "visible_pixels": visible_pixels,
        }
        item["ok"] = bool(
            settle_shift < 0.001
            and item["yaw_change_deg"] < 0.5
            and ws.up_z(name) > 0.9999
            and contacts == ["table"]
            and abs(pos[2] - h.objects[name].half_height) < 0.0015
            and in_table
            and visible_pixels > 30
        )
        report["objects"][name] = item
        report["ok"] &= item["ok"]

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            gap = (
                float(np.linalg.norm(ws.pos(a)[:2] - ws.pos(b)[:2]))
                - h.objects[a].footprint_radius
                - h.objects[b].footprint_radius
            )
            report["object_gaps_cm"][f"{a}/{b}"] = round(100 * gap, 3)
            report["ok"] &= gap > 0.001
    for name in names:
        gaps = []
        for container, container_spec in h.containers.items():
            gap = _gap_to_tray(
                container_spec,
                (*ws.pos(container)[:2], ws.yaw(container)),
                tuple(ws.pos(name)[:2]),
            ) - h.objects[name].footprint_radius
            gaps.append(gap)
            report["tray_gaps_cm"][f"{name}/{container}"] = round(100 * gap, 3)
        report["ok"] &= min(gaps, default=1.0) > 0.001

    seg.close()
    report["ok"] = bool(report["ok"])
    return report


def render_set(env: SO101TabletopEnv, spec, entries: list[dict], out_path: Path, tile_size: int) -> None:
    """Render the four unique offsets as front/wrist/overview rows."""
    renderer = mujoco.Renderer(env.model, tile_size, tile_size)
    overview = mujoco.MjvCamera()
    overview.lookat[:] = (0.13, 0.0, 0.0)
    overview.distance, overview.azimuth, overview.elevation = 0.70, 200.0, -60.0
    rows = []
    for entry in entries[: len(spec.offsets_m)]:
        env.reset(options={"init_state": entry["init_state"]})
        dx, dy = entry["offset_m"]
        title = f"{entry['offset_label']}: dx={100 * dx:+.0f} cm, dy={100 * dy:+.0f} cm"
        tiles = []
        for camera in ("front", "wrist_cam"):
            renderer.update_scene(env.data, camera=camera)
            tiles.append(renderer.render().copy())
        renderer.update_scene(env.data, camera=overview)
        tiles.append(renderer.render().copy())
        tiles[0] = label(tiles[0], f"{title} | front")
        tiles[1] = label(tiles[1], "wrist")
        tiles[2] = label(tiles[2], "overview (not a policy input)")
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_path)
    renderer.close()


def training_probe_offsets() -> list[tuple[str, tuple[float, float]]]:
    """Grid corners/edges/centre plus fixed interior probes for the training rectangle."""
    points = []
    for dx in (TRAIN_X_RANGE_M[0], 0.0, TRAIN_X_RANGE_M[1]):
        for dy in (TRAIN_Y_RANGE_M[0], 0.0, TRAIN_Y_RANGE_M[1]):
            points.append((f"grid_{dx:+.2f}_{dy:+.2f}", (dx, dy)))
    rng = np.random.default_rng(20261105)
    random_x = rng.uniform(*TRAIN_X_RANGE_M, size=16)
    random_y = rng.uniform(*TRAIN_Y_RANGE_M, size=16)
    for i, (dx, dy) in enumerate(zip(random_x, random_y)):
        points.append((f"random_{i:02d}", (float(dx), float(dy))))
    return points


def tag_counts(entries: list[dict]) -> dict[str, int]:
    counts = {"in-training-region": 0, "out-of-training-region": 0}
    for entry in entries:
        counts[entry["training_region_tag"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/residual/placements"))
    parser.add_argument("--tile-size", type=int, default=384)
    args = parser.parse_args()

    require_gpu_node()
    args.out.mkdir(parents=True, exist_ok=True)
    task = get_task("cube_flip")
    render_env = SO101TabletopEnv(task, EnvConfig(image_size=(256, 256), render=True))
    expert_env = SO101TabletopEnv(
        task,
        EnvConfig(render=False, terminate_on_success=False, max_steps=episode_limit(task.task_id)),
    )

    report: dict = {
        "task_id": task.task_id,
        "scene_hash": scene_hash(task),
        "rules_hash": rules_hash(task),
        "policy_render_backend": os.environ["MUJOCO_GL"],
        "training_distribution": {
            "dx_m": list(TRAIN_X_RANGE_M),
            "dy_m": list(TRAIN_Y_RANGE_M),
            "sampling": "independent uniform",
            "group_shift_objects": list(task.layout.placements[0]),
            "probes": [],
        },
        "evaluation_sets": {},
        "evaluation_group_summary": {},
        "all_ok": False,
    }

    try:
        for probe, (dx, dy) in training_probe_offsets():
            geometry = geometry_report(render_env, shifted_init_state(dx, dy))
            report["training_distribution"]["probes"].append(
                {"name": probe, "offset_m": [dx, dy], "geometry": geometry}
            )
            if not geometry["ok"]:
                raise RuntimeError(f"training placement probe {probe} at {(dx, dy)} failed geometry checks")

        validated: dict[str, list[dict]] = {}
        for spec in EVALUATION_SETS:
            entries = evaluation_entries(spec)
            unique_geometry = {}
            for label, offset in spec.offsets_m:
                geometry = geometry_report(render_env, shifted_init_state(*offset))
                unique_geometry[label] = {"offset_m": list(offset), "geometry": geometry}
                if not geometry["ok"]:
                    raise RuntimeError(f"{spec.name}/{label} failed geometry checks")

            expert_rows = []
            for entry in entries:
                ep = run_expert_episode(expert_env, entry["seed"], init_state=entry["init_state"])
                expert = {
                    "success": ep.success,
                    "failure": ep.failure,
                    "steps": ep.steps,
                    "success_step": ep.success_step,
                }
                entry["expert"] = expert
                expert_rows.append(
                    {
                        "index": entry["index"],
                        "seed": entry["seed"],
                        "offset_label": entry["offset_label"],
                        "training_region_tag": entry["training_region_tag"],
                        **expert,
                    }
                )
                print(
                    f"{spec.name} state {entry['index'] + 1:02d}/{spec.num_states}: "
                    f"{entry['offset_label']} seed={entry['seed']} success={ep.success} steps={ep.steps}",
                    flush=True,
                )
                if not ep.success:
                    report["evaluation_sets"][spec.name] = {
                        "description": spec.description,
                        "geometry": unique_geometry,
                        "experts": expert_rows,
                        "ok": False,
                    }
                    raise RuntimeError(
                        f"{spec.name} state {entry['index']} seed {entry['seed']} "
                        f"({entry['offset_label']}) failed expert check: {ep.failure}"
                    )

            validated[spec.name] = entries
            report["evaluation_sets"][spec.name] = {
                "description": spec.description,
                "geometry": unique_geometry,
                "experts": expert_rows,
                "training_region_counts": tag_counts(entries),
                "ok": True,
            }

        all_entries = [entry for entries in validated.values() for entry in entries]
        for tag in ("in-training-region", "out-of-training-region"):
            tagged = [entry for entry in all_entries if entry["training_region_tag"] == tag]
            report["evaluation_group_summary"][tag] = {
                "states": len(tagged),
                "expert_successes": sum(bool(entry["expert"]["success"]) for entry in tagged),
                "expert_failures": sum(not entry["expert"]["success"] for entry in tagged),
            }

        for spec in EVALUATION_SETS:
            entries = validated[spec.name]
            payload = {
                "task_id": task.task_id,
                "language": task.language,
                "layout": task.layout.name,
                "scene_hash": report["scene_hash"],
                "rules_hash": report["rules_hash"],
                "set_id": spec.name,
                "description": spec.description,
                "seed_base": spec.seed_base,
                "num": len(entries),
                "repeats_per_offset": spec.repeats,
                "training_region": {
                    "dx_m": list(TRAIN_X_RANGE_M),
                    "dy_m": list(TRAIN_Y_RANGE_M),
                    "inclusive": True,
                },
                "training_region_counts": tag_counts(entries),
                "rejected_candidates": [],
                "states": entries,
            }
            (args.out / f"{spec.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
            render_set(render_env, spec, entries, args.out / f"{spec.name}.png", args.tile_size)

        report["all_ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
        (args.out / "validation_report.json").write_text(json.dumps(report, indent=1) + "\n")
        raise
    finally:
        render_env.close()
        expert_env.close()

    (args.out / "validation_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"All placement and expert checks passed. Outputs: {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
