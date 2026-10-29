"""Inspect a three-cube task before any training: scenes, cameras, the expert, the success detectors, a dataset episode
and the full episode controller. Everything is rendered to files (headless server); the paths are printed.

    python -m so101_libero.cli.debug_three_cubes --task flip                      # all modes (replay if a dataset exists)
    python -m so101_libero.cli.debug_three_cubes --task sort layouts              # one mode
    python -m so101_libero.cli.debug_three_cubes --task pyramid expert controller --layout 3 --seed 7
    python -m so101_libero.cli.debug_three_cubes --task flip replay --repo-id local/smolvla_flip_red --episode 12
    bash scripts/debug_task.sh flip [modes...] [options]                          # the same through the runtime

Modes (outputs in --out, default outputs/debug/<task>):
  layouts     every placement (flip, pyramid: the 5 layouts; sort: 5 layouts x 5 starting permutations) reset and
              rendered: front camera (the policy's 256x256 view), wrist camera, an overview (not a policy input), in
              layouts.png (sort: also permutations.png, all 25 front views) and placements/*.png. Checks, printed and
              written to layouts_check.json: settled positions equal the nominal ones (< 1 mm, yaw < 0.5 deg), every
              cube upright and touching only the table, no robot contact at HOME, every cube inside the front image
              with >= 3 px margin and visible (segmentation pixels, >= 30), wrist-camera pixels (information), the gap
              between cube faces (> 0.5 mm), each cube's distance from the shoulder's pan axis, the clearance between
              the arm at HOME and the cubes (>= 1 cm, measured up to 5 cm), and the scripted expert solving the
              placement (physics only; --no-expert-check skips it).
  expert      one scripted demonstration from --layout (sort: --perm) with seed --seed, never stopped early: MP4 (front |
              wrist, captioned with the phase, the ground-truth success predicate and the real-robot behaviour detector),
              a key-frame sheet, and success_trace.csv: per control step the ingredients of both detectors (flip:
              rotation, top-face turn, shift, lift; sort: per-cube target errors; pyramid: green offset/height, contacts,
              base shift; behaviour: grasp-release cycles, gripper state, still steps, TCP height). Prints when each
              detector confirms.
  replay      open-loop replay of dataset episode --episode of --repo-id in the simulator from the episode's stored
              initial state: max deviation from the recorded joint states, goal at the end, mean pixel difference to
              the recorded (decoded) frames; MP4 of the replay next to the recorded frames.
  controller  the episode controller (so101_libero/control/) with the scripted expert as a stand-in for SmolVLA (served
              in virtual 50-action chunks) on a frozen evaluation state of --layout: state transitions with control-step
              numbers, success step, discarded queued actions, release / retreat / return-HOME results, MP4 with every
              frame labelled with the controller state, record JSON. Then the same episode with a "runaway" policy that
              keeps acting after its demonstration (it goes back down, grabs the cube it last placed and lifts it, as a
              memoryless policy that cannot see the task is done may do): with stop-on-success no runaway action is
              executed; with a fixed horizon instead (detector disabled) the runaway runs and the result is shown.
  all         layouts + expert + controller (+ replay when the dataset exists).
--viewer opens MuJoCo's interactive viewer at --layout if a display is available (not on a headless server).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MODES = ("layouts", "expert", "replay", "controller", "all")
# Default datasets (the naming of scripts/three_cubes_common.sh; scripts/debug_task.sh passes --repo-id explicitly).
DEFAULT_REPOS = {"flip_red_cube": "local/smolvla_flip_red", "sort_ryg": "local/smolvla_sort_ryg",
                 "build_pyramid": "local/smolvla_pyramid"}
LIST_NAMES = {"target_errors_cm": ("red", "yellow", "green"), "touching_robot": ("red", "yellow", "green"),
              "touch_red_yellow_table": ("red", "yellow", "table"), "base_shift_cm": ("red", "yellow")}
OVERVIEW = {"lookat": (0.17, 0.0, 0.0), "distance": 0.62, "azimuth": 180.0, "elevation": -55.0}


# ------------------------------------------------------------------------------------------------ helpers
def _task(name: str):
    from so101_libero.tasks.registry import get_task
    from so101_libero.tasks.three_cubes import TASK_ALIASES, resolve_task_id

    task = get_task(resolve_task_id(name))
    short = next((k for k, v in TASK_ALIASES.items() if v == task.task_id), task.task_id)
    return task, short


def _placement(task, layout: int, perm) -> int:
    """Placement id of a layout (sort: layout x starting permutation, given as letters like GRY or an index)."""
    lay = task.layout
    if not 0 <= layout < len(lay.layout_names):
        raise SystemExit(f"--layout must be in 0..{len(lay.layout_names) - 1}")
    if not lay.permutations:
        return int(layout)
    p = perm if perm is not None else 0
    k = lay.permutations.index(str(p).upper()) if not str(p).isdigit() else int(p)
    if not 0 <= k < len(lay.permutations):
        raise SystemExit(f"--perm must be one of {lay.permutations} or 0..{len(lay.permutations) - 1}")
    return len(lay.permutations) * int(layout) + k


def caption(frame: np.ndarray, lines: list[str], height: int = 14) -> np.ndarray:
    """``frame`` with a black caption bar of ``lines`` below it (even total height for H.264)."""
    from PIL import Image, ImageDraw

    h = len(lines) * height + 4
    h += h % 2
    bar = Image.new("RGB", (frame.shape[1], h), (0, 0, 0))
    d = ImageDraw.Draw(bar)
    for i, text in enumerate(lines):
        d.text((4, 2 + i * height), text, fill=(255, 255, 255))
    return np.concatenate([frame, np.asarray(bar)], axis=0)


def _sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    rows = [np.concatenate((tiles[i:i + cols] + [blank] * cols)[:cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def _save_png(img: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def _hull_area(pts: np.ndarray) -> float:
    """Area of the convex hull of 2-d points (monotone chain)."""
    pts = sorted(map(tuple, np.asarray(pts, dtype=float)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for seq in (pts, pts[::-1]):
        part = []
        for p in seq:
            while len(part) >= 2 and cross(part[-2], part[-1], p) <= 0:
                part.pop()
            part.append(p)
        hull += part[:-1]
    x, y = np.array(hull).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            names = LIST_NAMES.get(k)
            for i, x in enumerate(v):
                out[f"{prefix}{k}_{names[i] if names and len(names) == len(v) else i}"] = x
        elif isinstance(v, dict):
            continue
        else:
            out[f"{prefix}{k}"] = v
    return out


# ------------------------------------------------------------------------------------------------ layouts
def mode_layouts(task, short: str, out: Path, size: int, expert_check: bool) -> dict:
    """Render and check every placement of the task."""
    import mujoco

    from so101_libero.cli.render_placements import object_corners
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.sim.cameras import FRONT_CAMERA, project
    from so101_libero.utils.video import label

    lay = task.layout
    env = SO101TabletopEnv(task, EnvConfig(image_size=(256, 256)))
    m, d, h = env.model, env.data, env.h
    seg = {cam: mujoco.Renderer(m, 256, 256) for cam in ("front", "wrist_cam")}
    for r in seg.values():
        r.enable_segmentation_rendering()
    over = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = OVERVIEW["lookat"]
    cam.distance, cam.azimuth, cam.elevation = OVERVIEW["distance"], OVERVIEW["azimuth"], OVERVIEW["elevation"]
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in h.robot_bodies
                   and (m.geom_contype[g] or m.geom_conaffinity[g])]
    phys = None
    if expert_check:
        from so101_libero.cli.generate import episode_limit

        phys = SO101TabletopEnv(task, EnvConfig(render=False, terminate_on_success=False,
                                                max_steps=episode_limit(task.task_id)))
    names = list(h.objects)
    report = {"task_id": task.task_id, "layout": lay.name, "placements": []}
    tiles_by_pid = {}
    print(f"== layouts of {task.task_id}: {lay.num_placements} placements, renders in {out}")
    head = (f"{'placement':14s} {'settle':>6s} {'yaw':>5s} {'contacts':>8s} {'robot':>5s} {'margin':>6s} "
            f"{'front px':>8s} {'vis%':>4s} {'wrist px':>13s} {'gap':>6s} {'pan':>11s} {'HOME clr':>8s} {'expert':>10s}  result")
    print(head)
    for k in range(lay.num_placements):
        obs, _ = env.reset(options={"placement_id": k})
        info = lay.placement_info(k) if hasattr(lay, "placement_info") else {"placement_name": lay.placement_names[k]}
        ws = env.world
        r = {**info, "cubes": {}, "problems": [], "warnings": []}
        ids = {}
        for cname, rend in seg.items():
            rend.update_scene(d, camera=cname)
            ids[cname] = rend.render()[..., 0]
        for name in names:
            x, y, yaw = lay.placements[k][name]
            p = ws.pos(name)
            shift = float(np.linalg.norm(p[:2] - (x, y)))
            dyaw = abs((ws.yaw(name) - yaw + math.pi) % (2 * math.pi) - math.pi)
            sym = h.objects[name].yaw_symmetry  # a cube looks the same every 90 deg
            if sym:
                dyaw = min(dyaw % sym, sym - dyaw % sym)
            touching = sorted({m.geom(c.geom2 if c.geom1 in h.item_geoms[name] else c.geom1).name
                               for c in d.contact[:d.ncon] if c.geom1 in h.item_geoms[name] or c.geom2 in h.item_geoms[name]})
            uv = project(FRONT_CAMERA, object_corners(env, name), 256)
            margin = float(min(uv.min(), (256 - uv).min()))
            px = {c: int(np.isin(ids[c], list(h.item_geoms[name])).sum()) for c in ids}
            visible = px["front"] / max(1.0, _hull_area(uv))  # ~0.9-1.0 for an unhidden cube (edges)
            pan = float(np.linalg.norm(p[:2] - d.xpos[m.body("shoulder").id][:2]))
            r["cubes"][name] = {"pos": np.round(p, 4).tolist(), "settle_shift_mm": round(1e3 * shift, 3),
                                "yaw_change_deg": round(math.degrees(dyaw), 3), "up_z": round(ws.up_z(name), 5),
                                "contacts": touching, "front_margin_px": round(margin, 1),
                                "front_pixels": px["front"], "front_visible_fraction": round(visible, 3),
                                "wrist_pixels": px["wrist_cam"],
                                "pan_axis_distance_cm": round(100 * pan, 2)}
            if shift >= 0.001 or math.degrees(dyaw) >= 0.5:
                r["problems"].append(f"{name} moved while settling ({1e3 * shift:.2f} mm, {math.degrees(dyaw):.2f} deg)")
            if ws.up_z(name) < 0.9999 or touching != ["table"]:
                r["problems"].append(f"{name} not resting upright on the table only (contacts {touching})")
            if margin < 3.0:
                r["problems"].append(f"{name} within {margin:.1f} px of the front image border")
            if px["front"] < 30:
                r["problems"].append(f"{name} barely visible in the front image ({px['front']} px)")
            elif visible < 0.8:
                r["warnings"].append(f"{name} partly hidden in the front image at HOME ({100 * visible:.0f}% visible)")
            if px["wrist_cam"] == 0:
                r["warnings"].append(f"{name} outside the wrist camera's view at HOME")
        gaps = {f"{a}/{b}": float(mujoco.mj_geomDistance(m, d, next(iter(h.item_geoms[a])), next(iter(h.item_geoms[b])),
                                                          0.3, None)) for a, b in itertools.combinations(names, 2)}
        r["face_gaps_mm"] = {k2: round(1e3 * v, 2) for k2, v in gaps.items()}
        if min(gaps.values()) < 0.0005:
            r["problems"].append(f"cubes touching at the start ({min(gaps, key=gaps.get)})")
        robot_contacts = sorted({f"{m.body(m.geom_bodyid[c.geom1]).name}-{m.body(m.geom_bodyid[c.geom2]).name}"
                                 for c in d.contact[:d.ncon]
                                 if (m.geom_bodyid[c.geom1] in h.robot_bodies) != (m.geom_bodyid[c.geom2] in h.robot_bodies)})
        r["robot_contacts_at_home"] = robot_contacts
        if robot_contacts:
            r["problems"].append(f"robot contact at HOME: {robot_contacts}")
        cube_geoms = [g for n in names for g in h.item_geoms[n]]
        clr = min(mujoco.mj_geomDistance(m, d, cg, rg, 0.05, None) for cg in cube_geoms for rg in robot_geoms)
        r["home_clearance_cm"] = round(100 * float(clr), 2)
        if clr < 0.01:
            r["problems"].append(f"arm at HOME only {100 * clr:.2f} cm from a cube")
        if phys is not None:
            from so101_libero.expert.rollout import run_expert_episode

            ep = run_expert_episode(phys, 5_000 + k, placement_id=k)
            r["expert"] = {"success": ep.success, "failure": ep.failure, "steps": ep.steps,
                           "success_step": ep.success_step}
            if not ep.success:
                r["problems"].append(f"scripted expert fails here ({ep.failure})")
        r["ok"] = not r["problems"]
        report["placements"].append(r)
        cubes = r["cubes"].values()
        wrist = "/".join(str(c["wrist_pixels"]) for c in cubes)
        pans = "/".join(f"{c['pan_axis_distance_cm']:.1f}" for c in cubes)
        exp = (f"{'ok' if r['expert']['success'] else 'FAIL'} {r['expert']['steps']}" if "expert" in r else "-")
        print(f"{r['placement_name']:14s} {max(c['settle_shift_mm'] for c in cubes):5.2f}m "
              f"{max(c['yaw_change_deg'] for c in cubes):5.2f} {'table' if all(c['contacts'] == ['table'] for c in cubes) else 'OTHER':>8s} "
              f"{'none' if not robot_contacts else 'YES':>5s} {min(c['front_margin_px'] for c in cubes):6.1f} "
              f"{min(c['front_pixels'] for c in cubes):8d} {100 * min(c['front_visible_fraction'] for c in cubes):4.0f} "
              f"{wrist:>13s} {1e3 * min(gaps.values()):5.1f}m {pans:>11s} "
              f"{r['home_clearance_cm']:6.2f}cm {exp:>10s}  {'OK' if r['ok'] else 'FAIL: ' + '; '.join(r['problems'])}")
        for wmsg in r["warnings"]:
            print(f"{'':14s} note: {wmsg}")
        over.update_scene(d, camera=cam)
        tile = [label(obs["observation.images.front"], f"{r['placement_name']}: front"),
                label(obs["observation.images.wrist"], "wrist"), label(over.render().copy(), "overview (not a policy input)")]
        if size != 256:
            from PIL import Image

            tile = [np.asarray(Image.fromarray(t).resize((size, size))) if t.shape[0] != size else t for t in tile]
        tiles_by_pid[k] = tile
        _save_png(np.concatenate(tile, axis=1), out / "placements" / f"p{k:02d}_{r['placement_name'].replace('/', '_')}.png")
    if lay.permutations:  # one row per layout (first permutation) + all 25 front views
        n_perm = len(lay.permutations)
        rows = [np.concatenate(tiles_by_pid[n_perm * k], axis=1) for k in range(len(lay.layout_names))]
        _save_png(_sheet([tiles_by_pid[k][0] for k in range(lay.num_placements)], n_perm), out / "permutations.png")
        print(f"   all {lay.num_placements} front views (rows = layouts, columns = {', '.join(lay.permutations)}): "
              f"{(out / 'permutations.png').resolve()}")
    else:
        rows = [np.concatenate(tiles_by_pid[k], axis=1) for k in range(lay.num_placements)]
    _save_png(np.concatenate(rows, axis=0), out / "layouts.png")
    report["all_ok"] = all(p["ok"] for p in report["placements"])
    (out / "layouts_check.json").write_text(json.dumps(report, indent=1))
    print(f"   layouts: {'ALL OK' if report['all_ok'] else 'PROBLEMS (see above)'}  -> {(out / 'layouts.png').resolve()}, "
          f"{(out / 'layouts_check.json').resolve()}, {(out / 'placements').resolve()}/")
    for r in seg.values():
        r.close()
    over.close()
    env.close()
    if phys is not None:
        phys.close()
    return {"all_ok": report["all_ok"], "problems": [p for r in report["placements"] for p in r["problems"]]}


# ------------------------------------------------------------------------------------------------ expert
def mode_expert(task, short: str, out: Path, layout: int, perm, seed: int) -> dict:
    """One scripted demonstration, never stopped early, with both success detectors traced per step."""
    from so101_libero.cli.generate import episode_limit
    from so101_libero.control.completion import StepSignal, make_detector
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import expert_rng, make_expert
    from so101_libero.utils.video import write_mp4

    pid = _placement(task, layout, perm)
    env = SO101TabletopEnv(task, EnvConfig(image_size=(256, 256), terminate_on_success=False,
                                           max_steps=episode_limit(task.task_id)))
    env.reset(seed=seed, options={"placement_id": pid})
    name = task.layout.placement_names[pid]
    expert = make_expert(env, task.plan, expert_rng(seed))
    gt, beh = make_detector(task, "sim"), make_detector(task, "behavior")
    gt.reset()
    beh.reset()
    print(f"== expert demonstration of {task.task_id}, placement {name} (id {pid}), seed {seed}")
    stem = f"expert_{short}_{name.replace('/', '_')}_seed{seed}"
    frames, keyframes, rows, phases = [], [], [], []
    last_phase, step = None, 0

    def frame_now(phase: str) -> np.ndarray:
        g, b = gt.last, beh.last
        gt_txt = (f"ground truth: raw={g.get('raw', False)} streak={g.get('streak', 0)}/{gt.confirm_steps}"
                  + (f" CONFIRMED at {gt.confirmed_step}" if gt.confirmed_step else ""))
        beh_txt = (f"behaviour: cycles={b.get('cycles', 0)} {b.get('gripper_state', 'free')} still={b.get('still_steps', 0)}"
                   f" tcp_z={b.get('tcp_z', 0):.3f}" + (f" CONFIRMED at {beh.confirmed_step}" if beh.confirmed_step else ""))
        return caption(env.render(), [f"{task.task_id} {name} seed {seed} | step {step} | phase {phase}", gt_txt, beh_txt])

    frames.append(frame_now("start"))
    while True:
        action = expert.act()
        phase = expert.phase
        if action is None:
            break
        if phase != last_phase:
            phases.append((step, phase))
            keyframes.append(frame_now(phase))
            last_phase = phase
        _, _, _, _, info = env.step(action)
        step += 1
        q = env.joint_state()
        sig = StepSignal(step=step, t=step / env.cfg.control_hz, joints=q.astype(np.float64),
                         command=np.asarray(action, dtype=np.float64), world=env.world)
        gt_new = gt.update(sig) and gt.confirmed_step == step
        beh_new = beh.update(sig) and beh.confirmed_step == step
        rows.append({"step": step, "t_s": round(step / env.cfg.control_hz, 3), "phase": phase,
                     "gripper_cmd": round(float(action[5]), 2), "gripper_meas": round(float(q[5]), 2),
                     **_flatten("gt_", gt.last), "gt_confirmed": gt.confirmed_step is not None,
                     **_flatten("beh_", beh.last), "beh_confirmed": beh.confirmed_step is not None})
        frames.append(frame_now(phase))
        if gt_new or beh_new:
            keyframes.append(frame_now(f"{phase} [{'GROUND TRUTH' if gt_new else 'BEHAVIOUR'} CONFIRMED]"))
    final_ok = bool(task.check(env.world))
    keyframes.append(frame_now(f"end (goal holds: {final_ok})"))
    out.mkdir(parents=True, exist_ok=True)
    video = write_mp4(frames, out / f"{stem}.mp4", fps=env.cfg.control_hz)
    sheet = _save_png(_sheet(keyframes, 2), out / f"{stem}_keyframes.png")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "success_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {"task_id": task.task_id, "placement": name, "placement_id": pid, "seed": seed, "steps": step,
               "expert_failure": expert.failure, "goal_holds_at_end": final_ok,
               "ground_truth_confirmed_step": gt.confirmed_step, "behaviour_confirmed_step": beh.confirmed_step,
               "phases": phases, "video": str(video), "keyframes": str(sheet),
               "trace": str(out / "success_trace.csv")}
    (out / "expert_summary.json").write_text(json.dumps(summary, indent=1))
    print("   phases: " + ", ".join(f"{p}@{s}" for s, p in phases))
    print(f"   demonstration: {step} steps ({step / env.cfg.control_hz:.1f} s), expert failure: {expert.failure}, "
          f"goal holds at the end: {final_ok}")
    print(f"   ground-truth success confirmed at step {gt.confirmed_step} (raw predicate held {gt.confirm_steps} steps); "
          f"real-robot behaviour detector confirmed at step {beh.confirmed_step}"
          + (f" ({beh.confirmed_step - gt.confirmed_step:+d} steps)" if gt.confirmed_step and beh.confirmed_step else ""))
    print(f"   -> {video.resolve()}\n   -> {sheet.resolve()}\n   -> {(out / 'success_trace.csv').resolve()}")
    env.close()
    return summary


# ------------------------------------------------------------------------------------------------ replay
def _dataset_complete(root: Path) -> bool:
    """Generated to the end: the generator writes meta/so101_libero.json last (LeRobot's info.json exists from the
    start, so a dataset still being generated would otherwise look usable)."""
    return all((root / "meta" / f).exists() for f in ("info.json", "so101_libero.json", "so101_libero_episodes.jsonl"))


def mode_replay(task, short: str, out: Path, repo_id: str, episode: int) -> dict:
    """Open-loop replay of a dataset episode next to its recorded frames."""
    root = Path(os.environ.get("HF_LEROBOT_HOME", "data")) / repo_id
    if not _dataset_complete(root):
        state = "is still being written (or was interrupted)" if root.exists() else "does not exist"
        print(f"== replay: dataset {repo_id} at {root} {state}: skipped (collect it first: "
              f"bash scripts/collect_{short}.sh)")
        return {"skipped": True}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.tasks.registry import get_task
    from so101_libero.utils.video import label, write_mp4

    side = [json.loads(line) for line in open(root / "meta" / "so101_libero_episodes.jsonl")]
    if not 0 <= episode < len(side):
        raise SystemExit(f"--episode must be in 0..{len(side) - 1}")
    rec = side[episode]
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    a, b = int(ds.meta.episodes[episode]["dataset_from_index"]), int(ds.meta.episodes[episode]["dataset_to_index"])
    rows = ds.hf_dataset.with_format(None).select(range(a, b))
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    actions = np.asarray(rows["action"], dtype=np.float32)
    rtask = get_task(rec["task_id"])
    print(f"== replay of {repo_id} episode {episode}: {rtask.task_id}, placement {rec.get('placement_name')}, "
          f"seed {rec['seed']}, {len(actions)} frames")
    env = SO101TabletopEnv(rtask, EnvConfig(image_size=(256, 256), terminate_on_success=False, max_steps=100_000))
    obs, _ = env.reset(options={"init_state": rec["init_state"]})
    dev, pix, frames = 0.0, [], []
    info = {"is_success": False}

    def to_img(x) -> np.ndarray:
        return (x.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

    for t in range(len(actions)):
        dev = max(dev, float(np.max(np.abs(obs["observation.state"] - states[t]))))
        item = ds[a + t]
        rf, rw = to_img(item["observation.images.front"]), to_img(item["observation.images.wrist"])
        pix.append(float(np.abs(obs["observation.images.front"].astype(np.float32) - rf.astype(np.float32)).mean()))
        frames.append(np.concatenate([label(obs["observation.images.front"], f"replay t={t}"), label(rf, "recorded"),
                                      label(obs["observation.images.wrist"], "replay wrist"),
                                      label(rw, "recorded wrist")], axis=1))
        obs, _, _, _, info = env.step(actions[t])
    goal = bool(rtask.check(env.world))
    out.mkdir(parents=True, exist_ok=True)
    video = write_mp4(frames, out / f"replay_{Path(repo_id).name}_ep{episode:03d}.mp4", fps=env.cfg.control_hz)
    summary = {"repo_id": repo_id, "episode": episode, "task_id": rtask.task_id, "frames": len(actions),
               "max_state_deviation": dev, "success_streak_at_end": bool(info["is_success"]), "goal_holds_at_end": goal,
               "mean_abs_pixel_diff_front": round(float(np.mean(pix)), 3), "video": str(video)}
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"   replayed joint states vs recorded: max deviation {dev:.2e} (deg / gripper units); goal holds at the end: "
          f"{goal} (success confirmed: {info['is_success']}); front image vs recorded (AV1-decoded): mean |diff| "
          f"{np.mean(pix):.2f}/255\n   -> {video.resolve()}")
    env.close()
    return summary


# ------------------------------------------------------------------------------------------------ controller
class _RunawayMixin:
    """After its demonstration the policy keeps acting: back down to where it released the last cube with the jaws
    open, close them on the cube, and lift it again (then hold). Counts every ``act`` call and the runaway actions."""

    def reset(self) -> None:
        super().reset()
        self.calls = self.expert_actions = self.runaway_requested = 0
        self._release_cmd = None
        self._plan: list[np.ndarray] | None = None

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        self.calls += 1
        if self._served % self.chunk_size == 0:
            self.chunks += 1
        self._served += 1
        if self._plan is None:
            action = self.expert.act()
            if action is not None:
                if self.expert.phase == "release":  # where the last cube was put down (sort: the last one)
                    self._release_cmd = np.asarray(action, dtype=np.float64).copy()
                self.expert_actions += 1
                self._last = np.asarray(action, dtype=np.float32)
                return self._last.copy()
            self._plan = self._runaway_plan()
        self.runaway_requested += 1
        self._last = self._plan.pop(0) if self._plan else self._last
        return self._last.copy()

    def _runaway_plan(self) -> list[np.ndarray]:
        from so101_libero.control.home import min_jerk_path

        last = np.asarray(self._last, dtype=np.float64)
        down = (self._release_cmd if self._release_cmd is not None else last).copy()
        down[5] = 38.0  # jaws wide open, as when approaching a cube
        close = down.copy()
        close[5] = 0.0
        lift = last.copy()
        lift[5] = 0.0
        fps = 30.0
        path = [*min_jerk_path(last, down, fps, 30.0, 1.0), *min_jerk_path(down, close, fps, 60.0, 0.5),
                *min_jerk_path(close, lift, fps, 30.0, 1.0)]
        return [np.asarray(q, dtype=np.float32) for q in path]


def mode_controller(task, short: str, out: Path, layout: int, perm, trial: int) -> dict:
    """The episode controller with the expert as a chunked stand-in policy; then the runaway comparison."""
    from so101_libero.cli.run_episodes import episode_specs, run_sim_episode
    from so101_libero.control.policies import ExpertPolicy
    from so101_libero.control.robots import SimRobot

    specs = episode_specs(task, None, [layout])
    if task.layout.permutations:
        letters = task.layout.permutations[_placement(task, layout, perm) % len(task.layout.permutations)]
        specs = [s for s in specs if s.get("permutation") == letters]
    if not specs:
        raise SystemExit(f"no frozen evaluation state for layout {layout} (init_states/{task.task_id}.json)")
    spec = specs[min(trial, len(specs) - 1)]
    out.mkdir(parents=True, exist_ok=True)
    robot = SimRobot(task)
    sends = {"n": 0}
    send = robot.send

    def counting_send(target):
        sends["n"] += 1
        return send(target)

    robot.send = counting_send  # every command the arm receives, from the policy or the finisher
    events: list[tuple[int, str]] = []

    def log(msg: str) -> None:
        events.append((sends["n"], msg))

    print(f"== controller on {task.task_id}: frozen state {spec['index']} (layout {spec.get('layout_name')}"
          f"{' ' + spec['permutation'] if spec.get('permutation') else ''}, seed {spec['seed']}), the scripted expert "
          "as the policy (virtual 50-action chunks)")
    video = out / f"controller_{short}_state{spec['index']:03d}.mp4"
    rec, robot, policy = run_sim_episode(task, "expert", spec, robot=robot, video_path=video, log=log)
    for n, msg in events:
        if msg.startswith("[") or msg.startswith(("TASK_", "RETURN_HOME", "EPISODE_COMPLETE", "SAFE_", "discarded",
                                                  "STOP_POLICY")):
            print(f"   control step {n:4d}  {msg}")
    beh = next((dd for dd in rec.get("diagnostics", []) if dd.get("mode") == "behavior"), {})
    print(f"   success step {rec['success_step']} | queued actions discarded {rec['discarded_actions']} | chunks "
          f"{rec['chunks_predicted']} | release {rec['release'].get('needed')} | retreat {rec['retreat'].get('performed')}"
          f" | home error {rec['return_home'].get('max_error_deg')} deg | goal holds after HOME "
          f"{rec.get('goal_holds_after_home')} | episode complete {rec['episode_complete']}")
    print(f"   real-robot behaviour detector (diagnostic, not stopping anything): "
          + (f"confirmed at {beh['confirmed_step']}" if beh.get("confirmed_step") else
             "not confirmed before the ground truth stopped the run (it fires ~0.3-0.5 s later, during the final hold "
             "of a demonstration: see the expert mode)"))
    (out / "controller_record.json").write_text(json.dumps({**{k: v for k, v in spec.items() if k != "init_state"},
                                                            **rec}, indent=1, default=str))

    # ---- runaway policy: with stop-on-success, then with a fixed horizon (no success detection)
    class Runaway(_RunawayMixin, ExpertPolicy):
        pass

    runaway = {}
    for variant, kw in (("stop_on_success", {}), ("fixed_horizon", {"confirm_steps": 10 ** 9})):
        pol = Runaway(robot.env, seed=int(spec["seed"]))
        sends["n"] = 0
        events.clear()
        vpath = out / f"controller_{short}_runaway_{variant}.mp4" if variant == "fixed_horizon" else None
        r2, robot, _ = run_sim_episode(task, "expert", spec, robot=robot, policy=pol, video_path=vpath,
                                       behavior_diagnostic=False, log=log, **kw)
        policy_sends = next((n for n, msg in events if msg == f"[{'SUCCESS' if variant == 'stop_on_success' else 'TIMEOUT'}]"),
                            sends["n"])
        goal_at_stop = bool(r2.get("detector", {}).get("raw"))
        runaway[variant] = {"policy_steps": r2["policy_steps"], "act_calls": pol.calls,
                            "expert_actions": pol.expert_actions, "runaway_actions_executed": pol.runaway_requested,
                            "expert_finished_its_demo": bool(pol.expert.done), "termination": r2["termination"],
                            "commands_during_policy": policy_sends, "goal_holds_when_policy_stopped": goal_at_stop,
                            "goal_holds_after_home": r2.get("goal_holds_after_home"),
                            "release_needed": r2.get("release", {}).get("needed"),
                            "holding_when_stopped": r2.get("release", {}).get("holding_before"),
                            "discarded_actions": r2["discarded_actions"], "video": str(vpath) if vpath else None}
    s, f = runaway["stop_on_success"], runaway["fixed_horizon"]
    no_action_after_success = (s["act_calls"] == s["policy_steps"] == rec["success_step"]
                               and s["runaway_actions_executed"] == 0 and s["commands_during_policy"] == s["policy_steps"])
    print(f"   runaway policy (keeps acting after its demonstration: back down, grab the last cube, lift it):")
    print(f"     with stop-on-success: policy called {s['act_calls']} times = {s['policy_steps']} executed policy steps "
          f"(success at {rec['success_step']}); runaway actions executed: {s['runaway_actions_executed']}; its demo "
          f"was {'finished' if s['expert_finished_its_demo'] else 'NOT finished'} when stopped; {s['discarded_actions']} "
          f"queued actions discarded; goal after HOME: {s['goal_holds_after_home']}  -> "
          f"{'OK: no policy action after success' if no_action_after_success else 'FLAW: policy acted after success'}")
    print(f"     with a fixed horizon ({f['policy_steps']} steps, success detection off): runaway actions executed: "
          f"{f['runaway_actions_executed']}; goal holds when the policy stopped: {f['goal_holds_when_policy_stopped']} "
          f"(holding a cube: {f['holding_when_stopped']}, so the safe release opened the jaws: {f['release_needed']}); "
          f"after HOME: {f['goal_holds_after_home']}  -> {Path(f['video']).resolve()}")
    summary = {"record": str(out / "controller_record.json"), "video": str(video), "success_step": rec["success_step"],
               "discarded_actions": rec["discarded_actions"], "release": rec["release"], "retreat": rec["retreat"],
               "return_home": rec["return_home"], "goal_holds_after_home": rec.get("goal_holds_after_home"),
               "episode_complete": rec["episode_complete"], "behaviour_confirmed_step": beh.get("confirmed_step"),
               "runaway": runaway, "no_action_after_success": no_action_after_success}
    (out / "controller_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"   -> {video.resolve()}\n   -> {(out / 'controller_record.json').resolve()}\n   -> "
          f"{(out / 'controller_summary.json').resolve()}")
    robot.disconnect()
    return summary


# ------------------------------------------------------------------------------------------------ viewer
def open_viewer(task, layout: int, perm) -> None:
    if not os.environ.get("DISPLAY"):
        print("== viewer: headless server (no DISPLAY); look at the rendered PNGs / MP4s instead")
        return
    try:
        import mujoco.viewer

        from so101_libero.env import EnvConfig, SO101TabletopEnv

        env = SO101TabletopEnv(task, EnvConfig(render=False))
        env.reset(options={"placement_id": _placement(task, layout, perm)})
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            print("== viewer open (close the window to continue)")
            while v.is_running():
                v.sync()
                time.sleep(0.02)
    except Exception as e:  # noqa: BLE001  (no GLFW / display problems: the files are the fallback)
        print(f"== viewer unavailable ({type(e).__name__}: {e}); look at the rendered PNGs / MP4s instead")


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modes", nargs="*", default=None, metavar="MODE",
                    help=f"one or more of {', '.join(MODES)} (default all)")
    ap.add_argument("--task", required=True, help="flip | sort | pyramid (or a task id)")
    ap.add_argument("--out", type=Path, default=None, help="default outputs/debug/<task>")
    ap.add_argument("--layout", type=int, default=0, help="expert / controller / viewer: layout 0-4")
    ap.add_argument("--perm", default=None, help="sort: starting permutation, letters (GRY) or index 0-4 (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="expert: seed of the expert's timing/height randomness")
    ap.add_argument("--trial", type=int, default=0, help="controller: which frozen state of the layout (0-4)")
    ap.add_argument("--repo-id", default=None, help="replay: dataset (default local/smolvla_<task>)")
    ap.add_argument("--episode", type=int, default=0, help="replay: episode index")
    ap.add_argument("--size", type=int, default=256, help="layouts: tile size of the sheets")
    ap.add_argument("--no-expert-check", action="store_true", help="layouts: skip the expert run per placement")
    ap.add_argument("--viewer", action="store_true", help="open MuJoCo's viewer at --layout (needs a display)")
    args = ap.parse_args(argv)
    args.modes = args.modes or ["all"]
    bad = [m for m in args.modes if m not in MODES]
    if bad:
        ap.error(f"unknown mode(s) {bad}: choose from {', '.join(MODES)}")

    import so101_libero  # noqa: F401  (GL setup before mujoco)

    task, short = _task(args.task)
    out = args.out or Path("outputs/debug") / short
    repo = args.repo_id or DEFAULT_REPOS.get(task.task_id)
    modes = list(dict.fromkeys(args.modes))
    if "all" in modes:
        root = Path(os.environ.get("HF_LEROBOT_HOME", "data")) / (repo or "")
        modes = ["layouts", "expert", "controller"] + (["replay"] if repo and _dataset_complete(root) else [])
    print(f"debug {task.task_id} ('{task.language}'): modes {', '.join(modes)}; renderer "
          f"{os.environ.get('MUJOCO_GL')}; outputs in {out.resolve()}")
    results = {}
    t0 = time.time()
    if args.viewer:
        open_viewer(task, args.layout, args.perm)
    if "layouts" in modes:
        results["layouts"] = mode_layouts(task, short, out, args.size, not args.no_expert_check)
    if "expert" in modes:
        results["expert"] = mode_expert(task, short, out, args.layout, args.perm, args.seed)
    if "replay" in modes:
        results["replay"] = mode_replay(task, short, out, repo, args.episode)
    if "controller" in modes:
        results["controller"] = mode_controller(task, short, out, args.layout, args.perm, args.trial)
    (out / "debug_summary.json").write_text(json.dumps(results, indent=1, default=str))
    bad = []
    if "layouts" in results and not results["layouts"]["all_ok"]:
        bad.append("layouts")
    if "expert" in results and not results["expert"]["goal_holds_at_end"]:
        bad.append("expert")
    if "replay" in results and not results["replay"].get("skipped") and not (
            results["replay"]["goal_holds_at_end"] and results["replay"]["max_state_deviation"] < 1e-3):
        bad.append("replay")
    if "controller" in results and not (results["controller"]["episode_complete"]
                                        and results["controller"]["no_action_after_success"]):
        bad.append("controller")
    print(f"\ndebug {task.task_id}: {'all checks passed' if not bad else 'PROBLEMS in ' + ', '.join(bad)} "
          f"({time.time() - t0:.0f} s) -> {(out / 'debug_summary.json').resolve()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
