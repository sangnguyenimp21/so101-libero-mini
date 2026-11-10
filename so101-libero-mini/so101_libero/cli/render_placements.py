"""Render and check the predefined initial placements of a layout (so101_tri: 5 placements).

    python -m so101_libero.cli.render_placements --task cylinder_aside --out docs/media/tri_tasks [--expert-check]

For every placement: front camera (the policy's 256x256 view, upscaled), wrist camera and an overview (with the aside
zone outlined; the zone is only a region of the success check, nothing marks it in the scene), in one contact sheet
(placements.png), plus checks written to placements_check.json:
  - settled start positions equal the placement's nominal ones (no initial collision pushed anything), orientations
    unchanged, every object resting on the table with contacts only with the table;
  - spacing: footprint gaps between the objects and to the tray, distance from the shoulder's pan axis, clearance from
    the folded arm at rest and along the unfold to the home pose;
  - every object, the tray and the aside zone inside the 256x256 front image with a margin, and every object visible
    (segmentation pixels: not hidden behind the robot);
  - with --expert-check: each task of the layout solved by the scripted expert from this placement (physics only).
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


def layout_report(env, tri) -> tuple[dict, bool]:
    """Spacing of the current (reset) state in cm, and whether it meets the layout's requirements: footprint gaps
    between objects >= 5 cm and to the tray >= 3.5 cm, centres >= 17 cm from the pan axis, robot clearance >= 1.5 cm at
    rest and >= 1 cm along the unfold to the home pose (clearances are measured up to 5 cm)."""
    from so101_libero.sim.robot import HOME_QPOS
    from so101_libero.tasks.base import _gap_to_tray

    m, d, h, ws = env.model, env.data, env.h, env.world
    names = list(h.objects)
    out = {"object_gaps": {}, "tray_gaps": {}, "pan_axis_distance": {}}
    for a, b in itertools.combinations(names, 2):
        gap = float(np.linalg.norm(ws.pos(a)[:2] - ws.pos(b)[:2])) - h.objects[a].footprint_radius - h.objects[b].footprint_radius
        out["object_gaps"][f"{a}/{b}"] = round(100 * gap, 2)
    pan = h.model.body("shoulder").id
    mujoco.mj_kinematics(m, d)
    pan_xy = d.xpos[pan][:2].copy()
    for n in names:
        for c, spec in h.containers.items():
            gap = _gap_to_tray(spec, (*ws.pos(c)[:2], ws.yaw(c)), tuple(ws.pos(n)[:2])) - h.objects[n].footprint_radius
            out["tray_gaps"][n] = round(100 * gap, 2)
        out["pan_axis_distance"][n] = round(100 * float(np.linalg.norm(ws.pos(n)[:2] - pan_xy)), 2)
    # Clearance = MuJoCo's distance between robot and object geoms, searched up to 5 cm only: with a larger distmax,
    # mj_geomDistance (MuJoCo 3.3.7) returns spurious zeros for some distant mesh pairs (the green block).
    obj_geoms = [g for n in names for g in h.item_geoms[n] if m.geom_contype[g] or m.geom_conaffinity[g]]
    robot_geoms = [g for g in range(m.ngeom)
                   if m.geom_bodyid[g] in h.robot_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos

    def clearance(q, cap: float = 0.05):
        """Smallest distance (m) from any robot geom to any object geom in arm configuration ``q``, at most ``cap``."""
        scratch.qpos[h.joint_qpos_adr] = q
        mujoco.mj_kinematics(m, scratch)
        return min(cap, *(mujoco.mj_geomDistance(m, scratch, og, rg, cap, None) for og in obj_geoms for rg in robot_geoms))

    q0 = d.qpos[h.joint_qpos_adr].copy()
    q1 = np.r_[HOME_QPOS[:5], q0[5]]
    out["rest_clearance"] = round(100 * clearance(q0), 2)
    out["unfold_clearance"] = round(100 * min(clearance(q0 + s * (q1 - q0)) for s in np.linspace(0, 1, 13)), 2)
    ok = (min(out["object_gaps"].values()) >= 5.0 and min(out["tray_gaps"].values(), default=99) >= 3.5
          and min(out["pan_axis_distance"].values()) >= 17.0 and out["rest_clearance"] >= 1.5
          and out["unfold_clearance"] >= 1.0)
    return out, ok


def draw_zone(img: np.ndarray, model, cam: mujoco.MjvCamera, zone) -> np.ndarray:
    """Outline a table region (dashed) in an image of the free camera ``cam``."""
    if zone is None:
        return img
    from PIL import ImageDraw

    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    size = img.shape[0]
    f = (size / 2) / math.tan(math.radians(model.vis.global_.fovy) / 2)

    def px(p):
        v = np.asarray(p) - pos
        return (size / 2 + f * (v @ right) / (v @ fwd), size / 2 - f * (v @ up) / (v @ fwd))

    corners = [(zone.x[0], zone.y[0], 0.0), (zone.x[1], zone.y[0], 0.0), (zone.x[1], zone.y[1], 0.0), (zone.x[0], zone.y[1], 0.0)]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for a, b in zip(corners, corners[1:] + corners[:1]):
        pa, pb = np.array(px(a)), np.array(px(b))
        for t in np.linspace(0, 1, 13)[::2]:  # dashes
            dr.line([tuple(pa + (pb - pa) * t), tuple(pa + (pb - pa) * min(t + 1 / 12, 1))], fill=(0, 120, 255), width=3)
    lx, ly = px((zone.x[0], zone.y[1], 0.0))
    dr.text((lx + 4, ly + 4), "aside zone", fill=(0, 90, 220))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cylinder_aside", help="any task of the layout (the scene is shared)")
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
    overview.lookat[:] = (0.13, 0.0, 0.0)
    overview.distance, overview.azimuth, overview.elevation = 0.70, 200.0, -60.0
    m, d, h = env.model, env.data, env.h

    tri, check_spacing = None, False
    try:
        from so101_libero.tasks.cube_tasks import CUBES_LAYOUT
        from so101_libero.tasks.tri_tasks import TRI_LAYOUT, TRI_SCENE
        tri = TRI_SCENE if layout is TRI_LAYOUT else None
        check_spacing = layout is TRI_LAYOUT or layout is CUBES_LAYOUT
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
        if check_spacing:
            r["layout_cm"], layout_ok = layout_report(env, tri)
            ok &= layout_ok
        if tri is not None:
            zone = tri.aside_zone()
            zc = np.array([(x, y, z) for x in zone.x for y in zone.y for z in (0.0, 0.06)])
            uv = project(m, d, "front", zc, args.size)
            r["aside_zone_in_front_image"] = bool(np.all(uv >= args.margin_px) and np.all(uv <= args.size - args.margin_px))
            ok &= r["aside_zone_in_front_image"]
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
        print(json.dumps({"placement": k, "name": r["name"], "ok": r["ok"], "layout_cm": r.get("layout_cm"),
                          "expert": r.get("expert")}), flush=True)

        tiles = []
        for cam in ("front", "wrist_cam"):
            big.update_scene(d, camera=cam)
            tiles.append(big.render().copy())
        big.update_scene(d, camera=overview)
        tiles.append(draw_zone(big.render().copy(), m, overview, tri.aside_zone() if tri is not None else None))
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
