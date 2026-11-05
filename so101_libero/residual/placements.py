"""Deterministic shifted placements for residual-RL training and evaluation.

The original task and its committed ``init_states/cube_flip.json`` stay
unchanged.  A shifted state is a deep copy of the default cube-flip state in
which all three movable objects receive the same x/y translation.  Moving the
group preserves the original object spacing and orientations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

TRAIN_SEED_MIN = 20_000_000
TRAIN_SEED_MAX = 29_999_999
TRAIN_X_RANGE_M = (-0.015, 0.03)
TRAIN_Y_RANGE_M = (-0.02, 0.02)
_TRAIN_STREAM = 5_731


@dataclass(frozen=True)
class EvaluationSet:
    """One frozen evaluation set with four offsets and ten seeds per offset."""

    name: str
    description: str
    seed_base: int
    offsets_m: tuple[tuple[str, tuple[float, float]], ...]
    repeats: int = 10

    @property
    def num_states(self) -> int:
        return self.repeats * len(self.offsets_m)


EVALUATION_SETS = (
    EvaluationSet(
        name="shift_2cm_axis",
        description="Axis-aligned group shifts of 2 cm (within the training range).",
        seed_base=31_000_000,
        offsets_m=(
            ("forward", (0.02, 0.0)),
            ("backward", (-0.02, 0.0)),
            ("left", (0.0, 0.02)),
            ("right", (0.0, -0.02)),
        ),
    ),
    EvaluationSet(
        name="shift_1p5cm_diagonal",
        description="Diagonal group shifts at all (plus/minus 1.5 cm, plus/minus 1.5 cm) combinations.",
        seed_base=31_300_000,
        offsets_m=(
            ("forward_left", (0.015, 0.015)),
            ("forward_right", (0.015, -0.015)),
            ("backward_left", (-0.015, 0.015)),
            ("backward_right", (-0.015, -0.015)),
        ),
    ),
    EvaluationSet(
        name="shift_3cm_axis_feasible",
        description="Expert-feasible axis-aligned group shifts of 3 cm; backward is excluded as unreachable.",
        seed_base=31_100_000,
        offsets_m=(
            ("forward", (0.03, 0.0)),
            ("left", (0.0, 0.03)),
            ("right", (0.0, -0.03)),
        ),
    ),
    EvaluationSet(
        name="shift_4cm_axis_feasible",
        description="Expert-feasible axis-aligned group shifts of 4 cm; backward is excluded as unreachable.",
        seed_base=31_200_000,
        offsets_m=(
            ("forward", (0.04, 0.0)),
            ("left", (0.0, 0.04)),
            ("right", (0.0, -0.04)),
        ),
    ),
)


def default_init_state() -> dict:
    """Return a copy of the repository's frozen default cube-flip state."""
    from so101_libero.cli.make_init_states import load_init_states

    data = load_init_states("cube_flip")
    state = next(s for s in data["states"] if s["init_state"].get("placement_id") == 0)
    return deepcopy(state["init_state"])


def shifted_init_state(dx: float, dy: float, base_state: dict | None = None) -> dict:
    """Translate every movable object while preserving its yaw and robot pose."""
    state = deepcopy(default_init_state() if base_state is None else base_state)
    state.pop("placement_id", None)  # this is not one of the task's five named placements
    for pose in state["items"].values():
        pose[0] = float(pose[0] + dx)
        pose[1] = float(pose[1] + dy)
    return state


def training_offset(seed: int) -> tuple[float, float]:
    """Sample the continuous expert-feasible training rectangle reproducibly."""
    if not TRAIN_SEED_MIN <= seed <= TRAIN_SEED_MAX:
        raise ValueError(f"training seed {seed} is outside [{TRAIN_SEED_MIN}, {TRAIN_SEED_MAX}]")
    rng = np.random.default_rng([int(seed), _TRAIN_STREAM])
    dx = rng.uniform(*TRAIN_X_RANGE_M)
    dy = rng.uniform(*TRAIN_Y_RANGE_M)
    return float(dx), float(dy)


def training_init_state(seed: int) -> dict:
    """Return the reproducible continuous-offset state for one training episode."""
    return shifted_init_state(*training_offset(seed))


def training_region_tag(dx: float, dy: float, atol: float = 1e-12) -> str:
    """Classify an offset against the inclusive continuous training rectangle."""
    inside = (
        TRAIN_X_RANGE_M[0] - atol <= dx <= TRAIN_X_RANGE_M[1] + atol
        and TRAIN_Y_RANGE_M[0] - atol <= dy <= TRAIN_Y_RANGE_M[1] + atol
    )
    return "in-training-region" if inside else "out-of-training-region"


def evaluation_entries(spec: EvaluationSet) -> list[dict]:
    """Build balanced states, interleaving the four directions each repeat."""
    base = default_init_state()
    entries = []
    for repeat in range(spec.repeats):
        for label, (dx, dy) in spec.offsets_m:
            index = len(entries)
            entries.append(
                {
                    "seed": spec.seed_base + index,
                    "init_state": shifted_init_state(dx, dy, base),
                    "offset_label": label,
                    "offset_m": [dx, dy],
                    "training_region_tag": training_region_tag(dx, dy),
                    "repeat": repeat,
                    "index": index,
                }
            )
    return entries
