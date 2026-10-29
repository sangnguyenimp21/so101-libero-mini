"""Explicit execution logic of one episode, the same for simulated evaluation, simulated inference and the real arm:

    INITIALIZE -> MOVE_TO_HOME -> PREPARE_SCENE -> RUN_POLICY -> SUCCESS | TIMEOUT | ABORTED
               -> STOP_POLICY -> SAFE_RELEASE -> SAFE_RETREAT -> RETURN_HOME -> FINISHED

RUN_POLICY: per control step observe -> (completion check of the previous action) -> policy action -> checks -> send.
The detector is updated after EVERY executed action, before any new policy call; the moment it confirms completion the
loop ends: no further policy call, and the rest of the current action chunk is discarded (``policy.discard_pending``),
so the robot never keeps moving after the task is done just because a chunk or the horizon has actions left.
TIMEOUT = ``max_policy_steps`` executed (or ``max_seconds`` of wall time) without completion. ABORTED = a recoverable
runtime failure: camera / observation error, non-finite action, an action far outside the joint limits, a policy
exception, a communication error, a simulation failure (diverged, cube off the table), or the operator's Ctrl-C.

Every ending goes through the same deterministic finisher (so101_libero/control/episode_finisher.py): stop -> discard ->
release if needed -> retreat -> HOME. The arm does not move on its own after something went wrong: after an operator
interrupt the finisher runs only if ``confirm_finish`` allows it; after a communication error it is skipped; a finisher
stage that is interrupted or fails ends there (``finisher_interrupted``, the arm holds) and the caller must not start
another episode (run_episodes stops the session). On the real arm, an automatic MOVE_TO_HOME from a pose that is not
HOME needs ``confirm_move``. A dry run (``robot.dry_run``: nothing is sent) observes and infers, and skips every motion.

Success is reported for what the detector can actually judge (``success_basis``): "sim" = simulator ground truth;
"behavior" = the learned behaviour finished (TASK_SUCCESS is then logged as unverified); "vision" = behaviour finished
AND the front-camera verification at HOME passed (a failed verification is a failed episode). Logged separately (and
printed as event lines): TASK_SUCCESS (the manipulation, the primary metric), RETURN_HOME_SUCCESS, EPISODE_COMPLETE
(the arm is back at HOME, holding nothing, the finisher not interrupted).
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from so101_libero.control.completion.base import StepSignal
from so101_libero.control.home import home_error
from so101_libero.control.robots import RobotError

SUCCESS_BASIS = {"sim": "simulator ground truth", "behavior": "learned behaviour finished (result not verified)",
                 "vision": "learned behaviour finished + front-camera verification at HOME"}


class EpisodeState(str, Enum):
    INITIALIZE = "INITIALIZE"
    MOVE_TO_HOME = "MOVE_TO_HOME"
    PREPARE_SCENE = "PREPARE_SCENE"
    RUN_POLICY = "RUN_POLICY"
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"
    STOP_POLICY = "STOP_POLICY"
    SAFE_RELEASE = "SAFE_RELEASE"
    SAFE_RETREAT = "SAFE_RETREAT"
    RETURN_HOME = "RETURN_HOME"
    FINISHED = "FINISHED"


class EpisodeController:
    """Runs episodes of one task. ``robot``: control/robots.py backend; ``policy``: control/policies.py adapter;
    ``detector``: control/completion detector deciding completion; ``finisher``: EpisodeFinisher. ``diagnostics``:
    more detectors updated alongside without stopping anything (e.g. the real-robot behaviour detector in simulation,
    to measure how it agrees with the ground truth). ``on_frame(state, step)``: called once per control step (videos).
    ``confirm_finish(reason) -> bool``: may the arm move (release / retreat / HOME) after an operator interrupt?
    ``confirm_move(reason) -> bool``: real arm only, may it move to HOME from a pose that is not HOME?"""

    def __init__(self, robot, policy, detector, finisher, instruction: str, max_policy_steps: int,
                 max_seconds: float | None = None, diagnostics=(), on_frame=None, log=print,
                 limit_abort_margin_deg: float = 15.0, verify_frames: int = 5, confirm_finish=None,
                 confirm_move=None):
        self.robot, self.policy, self.detector, self.finisher = robot, policy, detector, finisher
        self.instruction = instruction
        self.max_policy_steps = int(max_policy_steps)
        self.max_seconds = max_seconds
        self.diagnostics = list(diagnostics)
        self.on_frame = on_frame
        self.log = log
        self.limit_abort_margin_deg = float(limit_abort_margin_deg)
        self.verify_frames = int(verify_frames)
        self.confirm_finish = confirm_finish or (lambda reason: True)
        self.confirm_move = confirm_move or (lambda reason: True)
        self.state = EpisodeState.INITIALIZE

    # --------------------------------------------------------------------------------------------- helpers
    def _enter(self, state: EpisodeState, record: dict, t0: float) -> None:
        self.state = state
        record["states"].append([state.value, record.get("policy_steps", 0), round(time.perf_counter() - t0, 3)])
        self.log(f"[{state.value}]")

    def _frame(self, step: int) -> None:
        if self.on_frame is not None:
            self.on_frame(self.state.value, step)

    def _check_action(self, action) -> str | None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (6,) or not np.all(np.isfinite(a)):
            return f"invalid action (non-finite or wrong shape): {np.round(a, 3).tolist()}"
        lim = self.robot.limits
        over = np.maximum(lim[:, 0] - a, a - lim[:, 1])
        if float(over[:5].max()) > self.limit_abort_margin_deg:
            j = int(np.argmax(over[:5]))
            return f"action {a[j]:.1f} on joint {j} is {over[j]:.1f} deg beyond its limit"
        return None

    @staticmethod
    def _check_obs(obs: dict) -> str | None:
        s = np.asarray(obs.get("observation.state", ()), dtype=np.float64)
        if s.shape != (6,) or not np.all(np.isfinite(s)):
            return "invalid observation.state"
        for k in ("observation.images.front", "observation.images.wrist"):
            img = obs.get(k)
            if img is None or np.asarray(img).ndim != 3:
                return f"invalid observation: {k} missing"
        return None

    @property
    def _dry_run(self) -> bool:
        return bool(getattr(self.robot, "dry_run", False))

    # --------------------------------------------------------------------------------------------- episode
    def run(self, episode: dict) -> dict:
        """One episode. ``episode``: index, seed, and for the simulator init_state or placement_id (see
        SimRobot.prepare_scene)."""
        mode = getattr(self.detector, "mode", "sim")
        rec = {"index": episode.get("index", 0), "seed": episode.get("seed"), "instruction": self.instruction,
               "states": [], "policy_steps": 0, "task_success": False, "success_basis": SUCCESS_BASIS.get(mode, mode),
               "completion_detected": False, "success_step": None, "task_completion_time_s": None,
               "termination": None, "discarded_actions": 0, "chunks_predicted": 0, "returned_home": False,
               "episode_complete": False, "error": None, "dry_run": self._dry_run}
        t0 = time.perf_counter()
        self._enter(EpisodeState.INITIALIZE, rec, t0)
        robot = self.robot

        # ---- MOVE_TO_HOME (the arm may be anywhere: folded at rest, or where the last episode ended)
        self._enter(EpisodeState.MOVE_TO_HOME, rec, t0)
        if robot.is_sim:
            rec["move_to_home"] = {"success": True, "note": "the simulator reset places the arm at HOME"}
        elif self._dry_run:
            rec["move_to_home"] = {"success": None, "note": "dry run: nothing is sent, the arm stays where it is"}
            self.log("MOVE_TO_HOME: dry run, nothing sent")
        else:
            try:
                err = home_error(robot.read_joints(), self.finisher.home)
                if err > self.finisher.cfg.home_tol_deg and not self.confirm_move(
                        f"the arm is {err:.0f} deg (largest joint) away from HOME"):
                    rec.update(termination="aborted: the operator declined the move to HOME")
                    return self._end(rec, t0, finish=False)
                rec["move_to_home"] = self.finisher.move_to_home()
            except (RobotError, KeyboardInterrupt) as e:
                rec.update(termination=f"aborted: move to HOME failed ({type(e).__name__}: {e})", error=str(e),
                           finisher_interrupted=f"MOVE_TO_HOME: {type(e).__name__}: {e}")
                return self._end(rec, t0, finish=False)
            if not rec["move_to_home"]["success"]:
                rec.update(termination="aborted: HOME not reached before the episode")
                return self._end(rec, t0, finish=False)

        # ---- PREPARE_SCENE (simulator: reset to the start state; real: the operator sets up the cubes)
        self._enter(EpisodeState.PREPARE_SCENE, rec, t0)
        try:
            info = robot.prepare_scene(episode)
        except KeyboardInterrupt:
            rec.update(termination="aborted: operator interrupt before the start")
            return self._end(rec, t0, finish=False)
        rec["scene"] = {k: v for k, v in (info or {}).items() if k in ("placement_id", "task_id")}
        if robot.is_sim:
            rec["start_home_error_deg"] = round(home_error(robot.read_joints(), self.finisher.home), 3)

        # ---- RUN_POLICY
        self.policy.reset()
        self.detector.reset(episode)
        for d in self.diagnostics:
            d.reset(episode)
        verifier = getattr(self.detector, "verifier", None)
        self._enter(EpisodeState.RUN_POLICY, rec, t0)
        t_policy = time.perf_counter()
        robot.set_rendering(True)
        ending, reason = EpisodeState.TIMEOUT, "max policy steps reached"
        step, last_cmd = 0, None
        try:
            while True:
                obs = robot.observe()  # the state right after the previous action
                bad = self._check_obs(obs)
                if bad:
                    ending, reason = EpisodeState.ABORTED, bad
                    break
                if step > 0:  # completion is judged after every executed action, before any new policy call
                    t = step / robot.fps if robot.is_sim else time.perf_counter() - t_policy
                    sig = StepSignal(step=step, t=t, joints=np.asarray(obs["observation.state"], dtype=np.float64),
                                     command=np.asarray(last_cmd, dtype=np.float64), obs=obs, world=robot.world)
                    for d in self.diagnostics:
                        d.update(sig)
                    if self.detector.update(sig):
                        ending, reason = EpisodeState.SUCCESS, "task completion confirmed"
                        break
                    if robot.failure:
                        ending, reason = EpisodeState.ABORTED, f"simulation: {robot.failure}"
                        break
                if step >= self.max_policy_steps:
                    break
                if self.max_seconds and time.perf_counter() - t_policy > self.max_seconds:
                    reason = f"max episode time {self.max_seconds:g} s reached"
                    break
                if step == 0 and verifier is not None:
                    verifier.capture_start(obs)
                self._frame(step)
                try:
                    action = self.policy.act(obs, self.instruction)
                except (KeyboardInterrupt, RobotError):
                    raise
                except Exception as e:  # noqa: BLE001  (policy exception: stop safely, report)
                    ending, reason = EpisodeState.ABORTED, f"policy exception: {type(e).__name__}: {e}"
                    break
                bad = self._check_action(action)
                if bad:
                    ending, reason = EpisodeState.ABORTED, bad
                    break
                last_cmd = robot.send(action)
                step += 1
                rec["policy_steps"] = step
        except KeyboardInterrupt:
            ending, reason = EpisodeState.ABORTED, "operator interrupt (Ctrl-C)"
        except RobotError as e:
            ending, reason = EpisodeState.ABORTED, f"robot error: {e}"
            rec["error"] = str(e)
        self._enter(ending, rec, t0)
        rec["termination"] = {"SUCCESS": "success", "TIMEOUT": "timeout"}.get(ending.value, f"aborted: {reason}")
        rec["termination_reason"] = reason
        rec["detector"] = self.detector.status()
        rec["diagnostics"] = [d.status() for d in self.diagnostics]
        rec["chunks_predicted"] = int(getattr(self.policy, "chunks", 0))
        if ending is EpisodeState.SUCCESS:
            rec["completion_detected"] = True
            rec["success_step"] = step
            rec["task_completion_time_s"] = round(step / robot.fps if robot.is_sim
                                                  else time.perf_counter() - t_policy, 3)
            when = f"at policy step {step} ({rec['task_completion_time_s']} s)"
            if mode == "vision":  # decided after the verification at HOME
                self.log(f"BEHAVIOR_COMPLETE {when}: the result is checked by the camera at HOME")
            else:
                rec["task_success"] = True
                self.log(f"TASK_SUCCESS {when}" + ("" if mode == "sim" else
                                                   " [unverified: the learned behaviour finished]"))
        else:
            self.log(f"TASK_{'TIMEOUT' if ending is EpisodeState.TIMEOUT else 'ABORTED'}: {reason}")
        finish = True
        if reason.startswith("operator interrupt") and not self.confirm_finish(reason):
            finish = False
        if rec["error"] and "command failed" in str(rec["error"]):
            finish = False  # the arm cannot be commanded: nothing more to do safely
        return self._end(rec, t0, finish=finish, verifier=verifier, mode=mode)

    def _end(self, rec: dict, t0: float, finish: bool, verifier=None, mode: str = "sim") -> dict:
        robot, fin = self.robot, self.finisher
        self._enter(EpisodeState.STOP_POLICY, rec, t0)
        rec["stop"] = fin.stop_policy(self.policy, rec.get("termination") or "not started")
        try:
            rec["discarded_actions"] = fin.discard_pending_actions(self.policy)
        except Exception as e:  # noqa: BLE001  (a policy that never ran)
            rec["discarded_actions"] = 0
            rec["stop"]["discard_error"] = str(e)
        if self._dry_run:
            self.log("dry run: no release / retreat / HOME (nothing is sent)")
            rec["finisher_skipped"] = "dry run"
            finish = False
        if not finish:
            if not self._dry_run:
                self.log("finisher skipped: the arm holds its last command")
                rec.setdefault("finisher_skipped", True)
            self._enter(EpisodeState.FINISHED, rec, t0)
            return self._summarize(rec, t0)
        recording = self.on_frame is not None
        robot.set_rendering(recording)
        counter = {"step": rec["policy_steps"]}

        def on_step():
            counter["step"] += 1
            self._frame(counter["step"])

        fin.on_step = on_step if recording else None

        def stage(state: EpisodeState, fn, key: str) -> bool:
            self._enter(state, rec, t0)
            try:
                rec[key] = fn()
                return True
            except (RobotError, KeyboardInterrupt) as e:  # the arm holds; the session must stop here
                rec[key] = {"error": f"{type(e).__name__}: {e}"}
                rec["finisher_interrupted"] = f"{state.value}: {type(e).__name__}: {e}"
                self.log(f"{state.value} interrupted ({type(e).__name__}: {e}); the arm holds its last command")
                return False

        ok = stage(EpisodeState.SAFE_RELEASE, fin.safe_release_if_needed, "release")
        ok = ok and stage(EpisodeState.SAFE_RETREAT, fin.safe_retreat, "retreat")
        ok = ok and stage(EpisodeState.RETURN_HOME, fin.return_home, "return_home")
        rec["returned_home"] = bool(ok and rec.get("return_home", {}).get("success"))
        if ok:
            held = robot.holding_object()
            rec["holding_at_home"] = held
            if held:
                self.log("WARNING: the gripper still holds an object at HOME")
        if rec["returned_home"]:
            self.log("RETURN_HOME_SUCCESS")
        if robot.is_sim:  # the finished arrangement after the finisher (did parking disturb it?)
            rec["goal_holds_after_home"] = bool(robot.task.check(robot.world))
        if verifier is not None:
            if rec["returned_home"]:
                try:
                    robot.set_rendering(True)
                    frames = []
                    for _ in range(max(1, self.verify_frames)):
                        robot.send(robot.last_sent)
                        frames.append(robot.observe())
                    vis_ok, vis = verifier.verify(frames)
                    rec["vision_verified"] = bool(vis_ok)
                    rec["vision"] = vis
                    self.log(f"VISION_VERIFICATION ({verifier.name}): {'PASS' if vis_ok else 'FAIL'} {vis}")
                except (RobotError, KeyboardInterrupt) as e:
                    rec["vision"] = {"error": f"{type(e).__name__}: {e}"}
                    rec["finisher_interrupted"] = f"VISION_VERIFICATION: {type(e).__name__}: {e}"
            else:
                rec["vision"] = {"error": "not at HOME: the result could not be checked"}
            if mode == "vision" and rec["completion_detected"]:
                rec["task_success"] = bool(rec.get("vision_verified"))
                self.log("TASK_SUCCESS (verified by the camera at HOME)" if rec["task_success"]
                         else "TASK_FAILED: the behaviour finished but the camera check did not pass")
        self._enter(EpisodeState.FINISHED, rec, t0)
        return self._summarize(rec, t0)

    def _summarize(self, rec: dict, t0: float) -> dict:
        rec["episode_complete"] = bool(rec["returned_home"] and not rec.get("holding_at_home")
                                       and not rec.get("finisher_interrupted"))
        rec["wall_time_s"] = round(time.perf_counter() - t0, 2)
        if rec["episode_complete"]:
            self.log("EPISODE_COMPLETE")
        self.log(f"episode {rec['index']}: task_success={rec['task_success']} ({rec['success_basis']}) "
                 f"returned_home={rec['returned_home']} episode_complete={rec['episode_complete']} "
                 f"({rec['termination']}, {rec['policy_steps']} policy steps, {rec['discarded_actions']} queued "
                 f"actions discarded)")
        return rec
