"""Tests of the real-robot evaluation (scripts/eval_real.py) without hardware: a fake launcher stands in for
scripts/real_smolvla.sh (same output lines, same per-tick action log, same SIGINT behaviour).
python -m pytest -q tests/test_eval_real.py -p no:cacheprovider"""

import csv
import json
import sys
import textwrap

import pytest

from scripts import eval_real as er

FAKE_ROLLOUT = textwrap.dedent('''
    """Fake lerobot-rollout: prints real_smolvla.sh's lines and writes rollout_debug.py's log.csv and frames."""
    import os, signal, sys, time
    from pathlib import Path

    mode = os.environ["FAKE_MODE"]
    debug = Path(os.environ["FAKE_DEBUG_ROOT"]) / f"{os.environ['RUN']}_{time.time_ns()}"
    print(f"debug frames and action log -> {debug}", flush=True)
    print("crop front=0,280,720,720 wrist=0,40,240,240 -> 256x256", flush=True)
    print(f"task: {os.environ['TASK']}", flush=True)
    print(f"policy env: {os.environ.get('POLICY')}", flush=True)
    if mode == "crash":
        print("Traceback (most recent call last):\\nConnectionError: could not connect on port", flush=True)
        sys.exit(1)
    if mode == "hang":  # never reaches the control loop
        time.sleep(60)
        sys.exit(0)
    stop = False
    def on_signal(*_):
        global stop
        stop = True
    signal.signal(signal.SIGINT, signal.SIG_IGN if mode == "ignore_sigint" else on_signal)
    debug.mkdir(parents=True)
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    with (debug / "log.csv").open("w") as f:
        f.write("tick,time_s," + ",".join(f"state_{j}" for j in joints) + ","
                + ",".join(f"action_{j}" for j in joints) + "\\n")
    from PIL import Image
    duration, t0, tick = float(os.environ["DURATION"]), time.time(), 0
    while not stop:
        t = time.time() - t0
        if t >= duration and mode != "ignore_sigint":
            print("INFO Duration limit reached (%.0fs)" % duration, flush=True)
            break
        # "return": out 60 deg and back by 1 s; otherwise out and staying out
        lift = -60 + (60 * min(t, 1.0 - t) / 0.5 if mode == "return" and t < 1.0 else (0 if mode == "return" else 60 * min(t / 0.5, 1)))
        state = [0, lift, 35, 85, 0, 0]
        if tick == 0:
            state = ["nan"] * 6  # the first tick can have no state yet
        with (debug / "log.csv").open("a") as f:
            f.write(f"{tick},{t:.3f}," + ",".join(str(v) for v in state) + "," + ",".join(str(v) for v in state) + "\\n")
        if tick % 10 == 0:
            Image.new("RGB", (64, 48), (tick % 255, 80, 160)).save(debug / f"frame_{tick:05d}.jpg")
        tick += 1
        time.sleep(1 / 30)
    print("INFO Returning robot to initial position before shutdown...", flush=True)
    print("INFO Rollout finished", flush=True)
''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A train dir with one fake checkpoint per task and a fake launcher; returns a helper to pick its mode."""
    train = tmp_path / "train"
    for task in er.TASKS:
        policy = train / task.run / "checkpoints" / "080000" / "pretrained_model"
        policy.mkdir(parents=True)
        for name in er.REQUIRED_FILES:
            (policy / name).write_text(json.dumps({"type": "smolvla"}) if name == "config.json" else "x")
        (policy / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": f"local/{task.task_id}"}}))
    (tmp_path / "fake_rollout.py").write_text(FAKE_ROLLOUT)
    launcher = tmp_path / "fake_real_smolvla.sh"
    launcher.write_text(f'exec "{sys.executable}" "{tmp_path / "fake_rollout.py"}"\n')
    monkeypatch.setenv("FAKE_DEBUG_ROOT", str(tmp_path / "debug"))
    monkeypatch.setenv("POLICY", "lerobot/smolvla_base")  # must not leak into the launcher
    monkeypatch.setattr(er, "RUN_GRACE_S", 0.5)
    monkeypatch.setattr(er, "SHUTDOWN_GRACE_S", 0.5)
    monkeypatch.setattr(er, "FORCE_GRACE_S", 0.5)

    class Fake:
        root, train_dir = tmp_path, train
        launcher_path = launcher

        @staticmethod
        def mode(m):
            monkeypatch.setenv("FAKE_MODE", m)

    return Fake


def test_return_to_start_detector():
    det = er.ReturnToStart()
    assert not det.update([0, -60, 35, 85, 0])  # the start pose
    assert not det.update([0, -50, 35, 85, 0])  # 10 deg away, but it has not left yet
    assert not det.update([0, -30, 35, 85, 0]) and not det.left  # 30 deg: not left
    assert not det.update([0, -15, 35, 85, 0]) and det.left  # 45 deg: left
    assert det.update([0, -50, 35, 85, 0])  # back within 20 deg
    det = er.ReturnToStart()
    for arm in ([0, -60, 35, 85, 0], [0, 0, 35, 85, 0], [0, -35, 35, 85, 0]):
        fired = det.update(arm)
    assert not fired  # 25 deg: not back yet
    assert det.update([5, -50, 35, 85, 0]) and det.max_distance == 60


def test_wilson_matches_the_simulator_evaluation():
    from so101_libero.cli.evaluate import wilson

    for k, n in ((0, 10), (3, 10), (10, 10), (7, 9)):
        assert er.wilson(k, n) == pytest.approx(wilson(k, n))


def test_action_log_tail_reads_complete_rows_only(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("tick,time_s,state_shoulder_pan,state_shoulder_lift,state_elbow_flex,state_wrist_flex,"
                   "state_wrist_roll,state_gripper\n0,0.0,nan,nan,nan,nan,nan,nan\n1,0.03,1,2,3,4,5,6\n2,0.07,1,2")
    tail = er.ActionLogTail(log)
    assert tail.read() == [(0.03, [1.0, 2.0, 3.0, 4.0, 5.0])]  # nan row skipped, partial row kept back
    with log.open("a") as f:
        f.write(",3,4,5,6\n")
    assert tail.read() == [(0.07, [1.0, 2.0, 3.0, 4.0, 5.0])]
    assert tail.read() == []


@pytest.mark.parametrize("mode, max_seconds, startup, ended_by, rc", [
    ("return", 10, 30, "returned_to_start", 0),
    ("timeout", 1.5, 30, "time_limit", 0),
    ("crash", 10, 30, "error", 1),
    ("hang", 10, 1.0, "watchdog", None),
    ("ignore_sigint", 0.5, 30, "watchdog", None),
])
def test_run_attempt_ends(fake, mode, max_seconds, startup, ended_by, rc):
    fake.mode(mode)
    trial = fake.root / "out" / "trials" / "ep000"
    a = er.run_attempt(er.TASKS_BY_ID["cube_flip"], "080000", trial, max_seconds=max_seconds, early_stop=True,
                       startup_timeout=startup, launcher=fake.launcher_path)
    assert a["ended_by"] == ended_by
    if rc is not None:
        assert a["returncode"] == rc
    log = (trial / "rollout.log").read_text()
    assert "task: Pick up the red cube and flip it onto its side." in log  # the training instruction was sent
    assert not list((fake.root / "debug").glob("*")), "the debug dir is moved into the trial"
    if mode == "return":
        assert 0.7 < a["returned_at_s"] < 1.6 and a["signals"] == ["SIGINT"] and 50 < a["max_arm_dev_deg"] <= 60
        assert "policy env: None" in log  # POLICY from the caller's environment is not passed on
        assert a["ticks"] > 20 and (trial / "actions.csv").is_file() and a["files"]["actions"] == "actions.csv"
        assert a["crop"] == "front=0,280,720,720 wrist=0,40,240,240 -> 256x256"
        if a["files"]["video"]:
            assert (trial / "policy_view.mp4").stat().st_size > 0
    elif mode == "timeout":
        assert a["returned_at_s"] is None and a["signals"] == [] and 1.2 < a["rollout_s"] <= 1.6
    elif mode in ("crash", "hang"):
        assert a["ticks"] == 0 and a["files"]["actions"] is None
    else:  # ignores SIGINT: forced
        assert a["signals"][:2] == ["SIGINT", "SIGTERM"]


def test_session_records_resumes_and_summarizes(fake, monkeypatch, capsys):
    fake.mode("return")
    answers = iter([
        "", "y", "clean flip",          # trial 1: success
        "", "r", "cube was misplaced",  # trial 2, first attempt: discarded
        "", "n", "",                    # trial 2: failure
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    args = ["--train-dir", str(fake.train_dir), "--tasks", "cube_flip", "--launcher", str(fake.launcher_path)]
    assert er.main(["run", *args, "--trials", "2", "--operator", "tester"]) == 0

    out = fake.train_dir / "smolvla_tri_cube_flip_100" / "eval_real" / "080000"
    r = json.loads((out / "results.json").read_text())
    assert r["overall"] == {"episodes": 2, "successes": 1, "success_rate": 0.5,
                            "wilson95": [round(x, 4) for x in er.wilson(1, 2)]}
    assert r["complete"] and r["protocol"]["operator"] == "tester" and r["render_backend"] == "real"
    assert r["policy"]["training_dataset"] == "local/cube_flip"
    t = r["per_task"]["cube_flip"]
    assert t["outcomes"] == {"success": 1, "failure": 1} and t["ended_by"] == {"returned_to_start": 2}
    e0, e1 = r["episodes"]
    assert (e0["index"], e0["success"], e0["note"], e0["dir"]) == (0, True, "clean flip", "trials/ep000")
    assert (e1["index"], e1["success"]) == (1, False) and (out / e1["log"]).is_file()
    assert (out / e0["actions"]).is_file()
    assert [d["reason"] for d in r["discarded"]] == ["cube was misplaced"]
    assert (out / r["discarded"][0]["dir"] / "rollout.log").is_file()
    rows = list(csv.DictReader((out / "episodes.csv").open()))
    assert [(row["index"], row["success"], row["note"]) for row in rows] == [("0", "True", "clean flip"),
                                                                              ("1", "False", "")]

    # Resume: one more trial planned; an unjudged leftover attempt dir is discarded first.
    (out / "trials" / "ep002").mkdir()
    answers = iter(["", "y", ""])
    assert er.main(["run", *args, "--trials", "3"]) == 0
    r = json.loads((out / "results.json").read_text())
    assert r["overall"]["episodes"] == 3 and r["overall"]["successes"] == 2 and len(r["sessions"]) == 2
    assert r["discarded"][-1]["reason"].startswith("interrupted session")

    # Complete: nothing to run, no input needed.
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked for input"))
    assert er.main(["run", *args, "--trials", "3"]) == 0
    capsys.readouterr()
    assert er.main(["summary", "--train-dir", str(fake.train_dir)]) == 0
    table = capsys.readouterr().out
    assert "| cube_flip | smolvla_tri_cube_flip_100 | 080000 | 3/3 | 2/3 = 67% |" in table
    assert "| green_push_forward | smolvla_tri_green_push_100 | 080000 | 0/? |" in table

    assert er.main(["push", "--train-dir", str(fake.train_dir), "--repo-owner", "someone", "--dry-run"]) == 0
    listing = capsys.readouterr().out
    assert "-> someone/smolvla_tri_cube_flip_100 (model) eval_real/080000/" in listing
    assert "results.json" in listing and "trials/ep000/actions.csv" in listing and ".jpg" not in listing


def test_quit_before_first_trial_writes_nothing_misleading(fake, monkeypatch):
    fake.mode("return")
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert er.main(["run", "--train-dir", str(fake.train_dir), "--tasks", "green_push_forward",
                    "--launcher", str(fake.launcher_path)]) == 0
    r = json.loads((fake.train_dir / "smolvla_tri_green_push_100" / "eval_real" / "080000" / "results.json").read_text())
    assert r["overall"]["episodes"] == 0 and not r["complete"] and r["episodes"] == []


def test_missing_requested_checkpoint_is_an_error(fake):
    assert er.main(["summary", "--train-dir", str(fake.train_dir), "--ckpt", "123456", "--tasks", "cube_flip"]) == 2
