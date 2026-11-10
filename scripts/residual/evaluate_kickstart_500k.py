"""Run the existing matched evaluator on the immutable kickstart 500k snapshot.

The evaluator requires best and comparison model paths. Both point to the
same 500k snapshot; the reported 'final' column is a duplicate, not the
running job's eventual final model. The residual environment uses the saved
kickstart scales, including its nonzero late gripper scale.
"""
import functools
import json
import sys
from pathlib import Path
from scripts.residual import evaluate_residual as evaluator
from so101_libero.residual.residual_env import ResidualConfig

RUN = Path('docs/residual/results/ppo_kickstart')
CHECKPOINT = RUN/'checkpoints/ppo_residual_500000_steps.zip'
OUT = Path('docs/residual/results/kickstart_500k_eval')
config = json.loads((RUN/'config.json').read_text())
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)

evaluator.ResidualConfig = functools.partial(
    ResidualConfig,
    early_arm_scale_deg=float(config['early_arm_scale']),
    late_arm_scale_deg=float(config['late_arm_scale']),
    early_gripper_scale=float(config['early_gripper_scale']),
    late_gripper_scale=float(config['late_gripper_scale']),
    use_cube_pose=not bool(config['no_cube_pose']),
)
sys.argv = ['evaluate_residual.py', '--model', str(CHECKPOINT),
            '--comparison-model', str(CHECKPOINT), '--train-dir', str(RUN),
            '--out', str(OUT), '--videos', '0']
evaluator.main()
