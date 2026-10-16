"""Record, in a dataset generated before 2026-10-17 01:45, how to regenerate it bit for bit with today's code.

Those datasets were made by an expert whose IK fallback seeds had the mirrored pan sign (fixed since; see
ExpertConfig.legacy_ik). With legacy_ik=True the current code reproduces them exactly (verified on
all 200 episodes of so101_mini_t1_200). This adds expert_config / code hashes / a source snapshot to
meta/so101_libero.json, which LeRobot does not read.

    bash scripts/env.sh python scripts/annotate_legacy_dataset.py /scratch/e1583535/datasets/lerobot/local/<name>
Only for datasets generated before 2026-10-17 01:45 (no expert_config / code_hash recorded at generation time).
"""

import dataclasses
import json
import math
import sys
from pathlib import Path

import so101_libero  # noqa: F401
from so101_libero.cli.generate import code_hash, snapshot_sources
from so101_libero.cli.make_init_states import rules_hash, scene_hash
from so101_libero.expert.pick_place import ExpertConfig
from so101_libero.tasks.registry import TASKS

force = "--force" in sys.argv
for root in [Path(a) for a in sys.argv[1:] if a != "--force"]:
    p = root / "meta" / "so101_libero.json"
    meta = json.loads(p.read_text())
    if ("expert_config" in meta or "code_hash" in meta) and not meta.get("annotated_note") and not force:
        sys.exit(f"{root} already has generation-time provenance: refusing to overwrite it (--force to override)")
    cfg = ExpertConfig(max_joint_speed=math.radians(75.0), legacy_ik=True)
    meta.update({
        "expert_config": dataclasses.asdict(cfg),
        "regenerate_with": "python -m so101_libero.cli.generate ... --legacy-expert (same seeds) "
                           "using meta/so101_libero_src.tar.gz",
        "code_hash": code_hash(),
        "scene_hashes": {t: scene_hash(TASKS[t]) for t in meta["tasks"]},
        "rules_hashes": {t: rules_hash(TASKS[t]) for t in meta["tasks"]},
        "annotated_note": "fields added after generation by scripts/annotate_legacy_dataset.py; the scene gained a "
                          "base collider afterwards that no demonstration touches (bit-identical regeneration)",
    })
    snapshot_sources(root / "meta" / "so101_libero_src.tar.gz")
    p.write_text(json.dumps(meta, indent=2))
    print("annotated", root)
