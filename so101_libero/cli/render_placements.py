"""Render and check the predefined initial placements of a layout (so101_tri: 5 placements).

    python -m so101_libero.cli.render_placements --task cube_left --out docs/media/tri_tasks [--expert-check]

For every placement: front camera (the policy's 256x256 view, upscaled), wrist camera and an overview, in one contact
sheet (placements.png), plus checks written to placements_check.json:
  - settled start positions equal the placement's nominal ones (no initial collision pushed anything), orientations
    unchanged, every object resting on the table with contacts only with the table;
  - spacing in the scene's own directions (green block 6 cm FRONT of the cube, cylinder 6 cm RIGHT of the green block);
  - every object (and the white box, and the cube's 5 cm-left destination) inside the 256x256 front image with a margin,
    and visible (segmentation pixels: not hidden behind the robot);
  - with --expert-check: each so101_tri task solved by the scripted expert from this placement (physics only).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import so101_libero  # noqa: F401  (GL setup before mujoco)
import mujoco
import numpy as np
from PIL import Image

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.utils.video import label


def project(model, data, cam: str, pts: np.ndarray, size: int) -> np.ndarray:
    """Pixel coordinates (u right, v down) of world points in a square ``size`` image of camera ``cam``."""
    c = model.camera(cam).id
    pos, rot = data.cam_xpos[c], data.cam_xmat[c].reshape(3, 3)
    f = (size / 2) / math.tan(math.radians(model.cam_fovy[c]) / 2)
    pc = (np.asarray(pts) - pos) @ rot  # camera frame: x right, y up, looking along -z
    return np.stack([size / 2 + f * pc[:, 0] / -pc[:, 2], size / 2 - f * pc[:, 1] / -pc[:, 2]], axis=1)


def box_corners(center: np.ndarray, rot: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.array([center + rot @ (np.array(s) * half) for s in itertools.product((-1, 1), repeat=3)])


def object_corners(env, name: str) -> np.ndarray:
    spec = env.h.objects[name]
    if spec.shape == "cylinder":
        half = np.array([spec.size[0], spec.size[0], spec.size[1]])
    elif spec.shape == "half_cylinder":
        half = np.array([spec.size[1], spec.size[0], spec.size[0] / 2])
    else:
        half = np.asarray(spec.size)
    return box_corners(env.world.pos(name), env.world.rot(name), half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cube_left", help="any task of the layout (the scene is shared)")
    ap.add_argument("--out", type=Path, default=Path("outputs/placements"))
    ap.add_argument("--size", type=int, default=256, help="policy image size (checks)")
    ap.add_argument("--sheet-size", type=int, default=384, help="tile size in the contact sheet")
    ap.add_argument("--margin-px", type=float, default=3.0)
    ap.add_argument("--expert-check", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from so101_libero.tasks.registry import TASKS, get_task

    task = get_task(args.task)
    layout = task.layout
    if not layout.num_placements:
        raise SystemExit(f"{args.task}: layout {layout.name} has no predefined placements")
    env = SO101TabletopEnv(task, EnvConfig(image_size=(args.size, args.size)))
    big = mujoco.Renderer(env.model, args.sheet_size, args.sheet_size)
    seg = mujoco.Renderer(env.model, args.size, args.size)
    seg.enable_segmentation_rendering()
    overview = mujoco.MjvCamera()
    overview.lookat[:] = (0.17, 0.06, 0.0)
    overview.distance, overview.azimuth, overview.elevation = 0.62, 205.0, -58.0
    m, d, h = env.model, env.data, env.h

    tri = None
    try:
        from so101_libero.tasks.tri_tasks import TRI_LAYOUT, TRI_SCENE
        tri = TRI_SCENE if layout is TRI_LAYOUT else None
    except ImportError:
        pass
    rows, report = [], {"task": args.task, "layout": layout.name, "image_size": args.size, "placements": []}
    for k in range(layout.num_placements):
        obs, info = env.reset(options={"placement_id": k})
        ws = env.world
        nominal = layout.placements[k]
        r = {"placement_id": k, "name": layout.placement_names[k], "objects": {}}
        ok = True
        for name, (x, y, yaw) in nominal.items():
            p = ws.pos(name)
            dev = float(np.linalg.norm(p[:2] - (x, y)))
            dyaw = abs((ws.yaw(name) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if h.objects[name].yaw_symmetry:
                dyaw = min(dyaw, abs(h.objects[name].yaw_symmetry - dyaw))
            touching = sorted({m.geom(c.geom2 if c.geom1 in h.item_geoms[name] else c.geom1).name
                               for c in d.contact[:d.ncon] if c.geom1 in h.item_geoms[name] or c.geom2 in h.item_geoms[name]})
            uv = project(m, d, "front", object_corners(env, name), args.size)
            inside = bool(np.all(uv >= args.margin_px) and np.all(uv <= args.size - args.margin_px))
            seg.update_scene(d, camera="front")
            ids = seg.render()[..., 0]
            pixels = int(np.isin(ids, list(h.item_geoms[name])).sum())
            o = {"pos": np.round(p, 4).tolist(), "settle_shift_mm": round(dev * 1e3, 2),
                 "yaw_change_deg": round(math.degrees(dyaw), 2), "up_z": round(ws.up_z(name), 5),
                 "height_mm": round(float(p[2]) * 1e3, 2), "nominal_height_mm": round(h.objects[name].half_height * 1e3, 2),
                 "contacts": touching, "in_front_image": inside,
                 "image_bbox_px": np.round([*uv.min(0), *uv.max(0)], 1).tolist(), "visible_pixels": pixels}
            o["ok"] = (dev < 0.001 and o["yaw_change_deg"] < 0.5 and o["up_z"] > 0.9999 and touching == ["table"]
                       and abs(p[2] - h.objects[name].half_height) < 0.0015 and inside and pixels > 30)
            ok &= o["ok"]
            r["objects"][name] = o
        if tri is not None:
            dirs = tri.directions
            pc, pg, py = (ws.pos(n)[:2] for n in ("red_cube", "green_block", "gray_cylinder"))
            r["spacing_mm"] = {"green_front_of_cube": round(float((pg - pc) @ dirs.vec("front")) * 1e3, 2),
                               "green_lateral_to_cube": round(float((pg - pc) @ dirs.vec("left")) * 1e3, 2),
                               "cylinder_right_of_green": round(float((py - pg) @ dirs.vec("right")) * 1e3, 2),
                               "cylinder_front_of_green": round(float((py - pg) @ dirs.vec("front")) * 1e3, 2)}
            sp = r["spacing_mm"]
            ok &= (abs(sp["green_front_of_cube"] - 60) < 0.5 and abs(sp["green_lateral_to_cube"]) < 0.5
                   and abs(sp["cylinder_right_of_green"] - 60) < 0.5 and abs(sp["cylinder_front_of_green"]) < 0.5)
            # the cube's destination (5 cm LEFT) must be in view as well
            dest = np.r_[pc + tri.cube_move * dirs.vec("left"), h.objects["red_cube"].half_height]
            uv = project(m, d, "front", box_corners(dest, ws.rot("red_cube"), np.asarray(h.objects["red_cube"].size)),
                         args.size)
            r["cube_destination_in_front_image"] = bool(np.all(uv >= args.margin_px) and np.all(uv <= args.size - args.margin_px))
            ok &= r["cube_destination_in_front_image"]
        for c in h.containers:
            spec = h.containers[c]
            half = np.array([spec.inner_size[0] + spec.wall_thickness, spec.inner_size[1] + spec.wall_thickness,
                             spec.wall_height / 2])
            uv = project(m, d, "front", box_corners(ws.pos(c) + np.r_[0, 0, spec.wall_height / 2], ws.rot(c), half), args.size)
            r.setdefault("containers_in_front_image", {})[c] = bool(np.all(uv >= 0) and np.all(uv <= args.size))
        if args.expert_check:
            from so101_libero.cli.generate import episode_limit
            from so101_libero.expert.rollout import run_expert_episode

            r["expert"] = {}
            for t in TASKS.values():
                if t.layout is not layout:
                    continue
                e = SO101TabletopEnv(t, EnvConfig(render=False, terminate_on_success=False, max_steps=episode_limit(t.task_id)))
                ep = run_expert_episode(e, 5_000 + k, placement_id=k)
                r["expert"][t.task_id] = {"success": ep.success, "failure": ep.failure, "steps": ep.steps}
                ok &= ep.success
        r["ok"] = bool(ok)
        report["placements"].append(r)
        print(json.dumps({"placement": k, "name": r["name"], "ok": r["ok"], "spacing_mm": r.get("spacing_mm"),
                          "expert": r.get("expert")}), flush=True)

        tiles = []
        for cam in ("front", "wrist_cam"):
            big.update_scene(d, camera=cam)
            tiles.append(big.render().copy())
        big.update_scene(d, camera=overview)
        tiles.append(big.render().copy())
        tiles[0] = label(tiles[0], f"placement {k}: {r['name']} | front camera")
        tiles[1] = label(tiles[1], "wrist camera")
        tiles[2] = label(tiles[2], "overview (not a policy input)")
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    Image.fromarray(sheet).save(args.out / "placements.png")
    report["all_ok"] = all(p["ok"] for p in report["placements"])
    (args.out / "placements_check.json").write_text(json.dumps(report, indent=1))
    print(f"all placements ok: {report['all_ok']}  ->  {args.out / 'placements.png'}, {args.out / 'placements_check.json'}")


if __name__ == "__main__":
    main()
