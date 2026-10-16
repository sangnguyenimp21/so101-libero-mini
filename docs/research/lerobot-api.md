# LeRobot 0.4.2: datasets, training and inference for a custom MuJoCo SO-101 sim

All findings come from the installed source at `L=/scratch/e1583535/virtualenvs/lerobot042-act/lib/python3.12/site-packages/lerobot`. "VERIFIED" means I ran it through `env.sh` on the login node (CPU). I installed nothing.

## 0. Three problems to know first
- **torchcodec does not work in this container.** Loading it fails with `RuntimeError: Could not load libtorchcodec … libavutil.so.59/58/57/56: cannot open shared object file`. The image has no system FFmpeg and no `ffmpeg` binary. `get_safe_default_codec()` still returns `"torchcodec"` whenever the module imports, so the default `video_backend` (for datasets and for `--dataset.video_backend`) fails on the first read. Always use `video_backend="pyav"` / `--dataset.video_backend=pyav`. PyAV decoding works (AV1 through libdav1d). VERIFIED.
- **Video encoding works.** PyAV 15.1.0 bundles FFmpeg 7.1. The `libsvtav1`, `h264` (libx264), `hevc` (libx265) and `libaom-av1` encoders all encoded and decoded a test video. VERIFIED.
- **Do not run `pip install "lerobot[smolvla]==0.4.2"`.** It would pull in numpy 2.x (details in §4).

## 1. Writing datasets
Signature (`lerobot_dataset.py:1532`):
`LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None, use_videos=True, tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0, video_backend=None, batch_encoding_size=1)`

**Root and overwriting**
- `root` defaults to `$HF_LEROBOT_HOME/<repo_id>`, which `env.sh` sets to `/scratch/e1583535/datasets/lerobot/<repo_id>`.
- `create()` calls `root.mkdir(exist_ok=False)`, so it raises `FileExistsError` if the folder exists. VERIFIED.
- There is no overwrite flag: run `shutil.rmtree(root)` first.
- To append to an existing dataset, open it with `LeRobotDataset(repo_id, root=root)`. `lerobot-edit-dataset` supports `delete_episodes`, `split`, `merge` and `remove_feature`.

**Codec and file sizes are not settable through `create()`**
- Encoding is hardcoded to `encode_video_frames(img_dir, path, fps, overwrite=True)`: `libsvtav1`, `yuv420p`, `g=2`, `crf=30`, preset 12. `info.json` records `video.codec: av1`.
- File sizes are set afterwards with `ds.meta.update_chunk_settings(chunks_size=1000, data_files_size_in_mb=100, video_files_size_in_mb=200)` (defaults shown).

**`features` rules**
- Names must not contain `/`.
- `timestamp`, `frame_index`, `episode_index`, `index` and `task_index` are added automatically.
- Camera `names` must be exactly `["height","width","channels"]`. `dataset_to_policy_features` only converts the shape to CHW when `names[2]` is in `{"channel","channels"}`.

**What `add_frame(frame)` accepts (all VERIFIED)**
- `"task": str` is required; leaving it out raises `Missing features: {'task'}`.
- Do not pass `"timestamp"`. Validation runs before it is popped, so it raises `Extra features: {'timestamp'}`. The timestamp is set automatically to `frame_index/fps`.
- Vectors must be `np.ndarray` with the exact dtype and shape; float64 is rejected. Torch tensors are accepted (converted with `.numpy()`).
- Images: HWC uint8 numpy is accepted (PIL and float in [0,1] also work). CHW is rejected when the feature is declared `(H,W,3)`.
- Frames are written as PNG under `root/images/…`. This is synchronous unless writer threads or processes are > 0; `lerobot-record` uses 4 threads per camera.

**What `save_episode(episode_data=None, parallel_encoding=True)` does**
- Computes per-episode stats.
- Appends rows through an open `ParquetWriter`.
- Encodes each camera in its own process, appends the result into `videos/<key>/chunk-000/file-000.mp4`, then deletes the PNGs.
- To throw away the current episode instead, call `clear_episode_buffer()`.

**`finalize()` is required before reading.** It writes the parquet footers for `data/` and `meta/episodes/`; episode metadata is buffered 10 episodes at a time. With `batch_encoding_size>1` the last episodes also still need encoding. `VideoEncodingManager` (in `video_utils.py`) does both, the same way `lerobot-record` uses it.

**Files on disk after a 2-episode test (VERIFIED)**
- `data/chunk-000/file-000.parquet`
- `meta/info.json`: totals, fps, features, per-video info.
- `meta/stats.json`: min/max/mean/std/count/q01/q10/q50/q90/q99 per feature; image stats have shape (3,1,1).
- `meta/tasks.parquet`: the task string is the index, with a `task_index` column.
- `meta/episodes/chunk-000/file-000.parquet`: `episode_index`, `tasks`, `length`, `data/chunk_index`, `data/file_index`, `dataset_from_index`/`dataset_to_index`, `videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}`, `stats/*`.

```python
import shutil, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
names = [f"{j}.pos" for j in ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]]
cam = lambda: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": names},
            "action": {"dtype": "float32", "shape": (6,), "names": names},
            "observation.images.front": cam(), "observation.images.wrist": cam()}
shutil.rmtree(root, ignore_errors=True)
ds = LeRobotDataset.create("local/so101_sim", fps=30, features=features, root=root,
                           robot_type="so101_follower", image_writer_threads=8, video_backend="pyav")
with VideoEncodingManager(ds):                      # encodes leftovers + ds.finalize()
    for ep in episodes:
        for st, act, front, wrist in ep:            # float32[6], float32[6], uint8 HxWx3 x2
            ds.add_frame({"observation.state": st, "action": act, "observation.images.front": front,
                          "observation.images.wrist": wrist, "task": task_str})
        ds.save_episode()
ds.stop_image_writer()
```

## 2. Reading datasets
Signature: `LeRobotDataset(repo_id, root=None, episodes=None, image_transforms=None, delta_timestamps=None, tolerance_s=1e-4, revision=None, force_cache_sync=False, download_videos=True, video_backend=None, batch_encoding_size=1)`

- **Backends:** `"torchcodec"` (broken here), `"pyav"` (works), `"video_reader"` (torchvision; UNVERIFIED).
- **Offline loading:** a complete local `root` loads offline. If any data or video file is missing, it silently tries a hub download, which fails while `HF_HUB_OFFLINE=1`.

**Item format (VERIFIED)**
- Video keys: float32 `(3,H,W)` in [0,1].
- `observation.state` and `action`: float32 `(6,)`.
- `timestamp`: float32 `()`.
- `frame_index`, `episode_index`, `index`, `task_index`: int64 `()`.
- `task`: `str`.

**`delta_timestamps` (VERIFIED)**
- `{"action": [i/30 for i in range(10)], "observation.images.front": [-1/30, 0]}` gives `action (10,6)` plus `action_is_pad` (bool, 10) and `front (2,3,H,W)`.
- Indices outside the episode are clamped to the episode boundary.
- `lerobot-train` builds these from the policy's `*_delta_indices`.

**Metadata fields:** `ds.meta.fps`, `.features`, `.total_episodes`, `.total_frames`, `.total_tasks`, `.tasks` (DataFrame), `.episodes` (HF Dataset), `.stats` (numpy), `.camera_keys`/`.video_keys`, `.robot_type`; also `ds.num_episodes` and `ds.num_frames`.

## 3. Real SO-101 conventions
**Motors (`robots/so101_follower`)**
- `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` (ids 1–5) use `MotorNormMode.RANGE_M100_100`, or `DEGREES` if `use_degrees=True` (default `False`).
- `gripper` (id 6) always uses `RANGE_0_100`.

**Formulas (`motors/motors_bus.py:770`)**
- M100_100: `(clip(raw)-min)/(max-min)*200-100`.
- 0_100: `…*100`.
- DEGREES: `(raw-mid)*360/4095`, unclipped.

**What `lerobot-record` writes**
- `observation.state` and `action` are float32 `(6,)` with names `[shoulder_pan.pos, …, gripper.pos]`.
- Cameras become `observation.images.<name>`, stored as video `(H,W,3)`.
- Example camera names in the CLI help: `front` (teleoperate, robot_client) and `laptop` (record).

**The user's existing datasets:** `/scratch/e1583535/datasets/lerobot/local/so101_office3_cube` and `so101_liftcube_mujoco` match this layout exactly. They use `front` + `wrist` at 480×640 AV1, 30 fps, joint values about ±100 and gripper about 6–57, so they are in RANGE mode, not degrees. Scale sim joint radians the same way if you want compatibility with real data or `smolvla_base`.

## 4. Training
This command ran for 2 steps on CPU in about 30 s and produced checkpoints. VERIFIED:
```
lerobot-train --dataset.repo_id=local/so101_sim --dataset.root=$ROOT --dataset.video_backend=pyav \
 --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=$OUT \
 --steps=100000 --batch_size=8 --save_freq=20000 --log_freq=200 --num_workers=4 --wandb.enable=false
```

**CLI facts**
- `--dataset.repo_id` is required but can be any `ns/name`.
- `policy.push_to_hub` defaults to **True**. Without `--policy.push_to_hub=false` (or a `--policy.repo_id`), `validate()` raises `'policy.repo_id' argument missing`.
- `--output_dir` must not exist, unless you pass `--resume=true --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json`.
- Defaults: steps 100000, batch_size 8, num_workers 4, save_freq 20000, log_freq 200, eval_freq 20000, seed 1000.
- `--dataset.use_imagenet_stats=true` by default, so camera stats are replaced with ImageNet mean/std.
- `--dataset.image_transforms.enable=false` by default.
- `--policy.device` defaults to automatic selection.
- **Evaluation during training** only runs when `--env.type` is given and `eval_freq>0`. Leave out `--env.*` (or pass `--eval_freq=0`) to disable it.
- **Checkpoints:** `$OUT/checkpoints/<000000-style step>/pretrained_model/` holds `config.json`, `model.safetensors`, `train_config.json`, `policy_preprocessor.json`, `policy_preprocessor_step_3_normalizer_processor.safetensors`, `policy_postprocessor.json` and `policy_postprocessor_step_0_unnormalizer_processor.safetensors`. There is also `training_state/` and a `checkpoints/last` symlink.

**ACTConfig defaults**
- chunk_size 100, n_action_steps 100, n_obs_steps 1.
- MEAN_STD normalization for VISUAL, STATE and ACTION.
- Backbone resnet18 with `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`.
- dim_model 512, 8 heads, ff 3200, 4 encoder / 1 decoder layers.
- VAE on (latent 32, kl_weight 10), dropout 0.1.
- AdamW lr 1e-5, weight decay 1e-4, backbone lr 1e-5, no scheduler. 52M parameters.
- Vision tokens come from the layer4 map: 8×8 per camera at 256 px.
- Temporal ensembling needs `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`.

**ACT ImageNet weights:** `resnet18-f37072fd.pth` (46.8 MB) is already in `/scratch/e1583535/cache/torch/hub/checkpoints`, and nothing was downloaded (VERIFIED). `from_pretrained` rebuilds the backbone, so `TORCH_HOME` is also needed at inference.

**DiffusionConfig defaults**
- n_obs_steps 2, horizon 16, n_action_steps 8, drop_n_last_frames 7.
- crop_shape (84,84): random crop in training, center crop at eval.
- resnet18 trained from scratch with GroupNorm, 32 spatial-softmax keypoints, one encoder shared by all cameras. All cameras must have identical shapes, and `observation.state` is required.
- down_dims (512,1024,2048), 266.6M parameters (VERIFIED).
- DDPM with 100 training timesteps; `num_inference_steps=None` means 100 steps at inference.
- STATE and ACTION use MIN_MAX normalization; Adam lr 1e-4 with cosine schedule and 500 warmup steps.

**Diffusion with 256×256 images:** `validate_features` passes, because it only checks that the crop fits inside the image. So nothing crashes, but the 84×84 crop is about 11% of the image. Use `--policy.crop_shape="[240, 240]"` or `=null` (both parse; VERIFIED), or render images of about 96 px.

**SmolVLAConfig defaults**
- It needs `transformers`: `modeling_smolvla` currently fails with `No module named 'transformers'`.
- Base VLM: `vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`, `num_vlm_layers=16`.
- `load_vlm_weights=False` by default. With `--policy.type=smolvla` the VLM is therefore randomly initialised; its config and processor are still fetched from the hub.
- chunk_size 50, n_action_steps 50, num_steps 10.
- `max_state_dim`/`max_action_dim` 32 (zero-padded).
- `resize_imgs_with_padding=(512,512)`, then pixels are mapped to [-1,1].
- `empty_cameras=0`; each extra one adds a −1-filled masked image.
- VISUAL normalization IDENTITY; state and action MEAN_STD.
- Vision encoder frozen and only the action expert trained (`freeze_vision_encoder` and `train_expert_only` both True).
- AdamW lr 1e-4, cosine schedule (warmup 1000, decay over 30000 steps to 2.5e-6).

**How SmolVLA gets the language task:** the dataset item's `task` goes through a step that appends `"\n"`, then the tokenizer step (max length 48), which produces `observation.language.tokens` and `observation.language.attention_mask`.

**Minor SmolVLA bug:** `forward` reads `batch.get("actions_id_pad")` (typo), so padded actions are never masked out of the loss.

**SmolVLA fine-tune flags (UNVERIFIED end-to-end):**
`--policy.path=lerobot/smolvla_base --policy.push_to_hub=false --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'`
- The rename map parses (VERIFIED).
- The base model expects `observation.state[6]`, `observation.images.camera1/2/3 [3,256,256]` and `action[6]`. Its config sets `load_vlm_weights=true` and `push_to_hub=true`.
- A missing `camera3` is simply skipped.
- Without the rename map, `validate_visual_features_consistency` raises.

**SmolVLA cache:** neither `lerobot/smolvla_base` (907 MB) nor SmolVLM2-500M (2030 MB) is in `/scratch/e1583535/cache/hub` or `~/.cache`. Both are needed, and must be pre-downloaded with `HF_HUB_OFFLINE=0`.

**pip dry-run results (VERIFIED, nothing installed)**
- **As the container is set up:** it fails with ResolutionImpossible. The container sets `PIP_CONSTRAINT=/etc/pip/constraint.txt`, which pins `packaging==23.2`; lerobot needs `>=24.2`.
- **With `unset PIP_CONSTRAINT`:**
  - Pip says it "Would install docopt-0.6.2 num2words-0.5.14 numpy-2.5.4 rerun-sdk-0.26.2 tokenizers-0.22.2 transformers-4.57.6".
  - numpy 2.5.4 would sit in front of the container's numpy 1.26.4, because the venv uses `--system-site-packages`. It comes in via rerun-sdk, which requires `numpy>=2`.
- **Safer option:** `bash scripts/env.sh bash -c 'unset PIP_CONSTRAINT; pip install "transformers>=4.53.0,<5.0.0" "num2words>=0.5.14,<0.6.0"'`.
  - This would install only docopt-0.6.2, num2words-0.5.14, tokenizers-0.22.2 and transformers-4.57.6.
  - If the constraint is left on, it also downgrades `packaging` to 23.2.
  - Whether transformers 4.57.6 actually works with SmolVLA at runtime is UNVERIFIED.

## 5. Inference
**Loading a checkpoint**
- `PreTrainedPolicy.from_pretrained(path, *, config=None, …, strict=False)` builds `cls(config)`, loads `model.safetensors` onto `config.device` and calls `.eval()`.
- `get_policy_class("act"|"diffusion"|"smolvla")` returns the policy class.
- `make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None)` needs exactly one of `ds_meta` or `env_cfg`.

**Processor pipelines**
- `make_pre_post_processors(policy_cfg, pretrained_path=None, **kw)` accepts `preprocessor_overrides`, `postprocessor_overrides`, `dataset_stats` and the `*_config_filename` options.
- Given a `pretrained_path`, it calls `PolicyProcessorPipeline.from_pretrained(path, "policy_preprocessor.json", overrides=…)`, which rebuilds each registered step and loads its safetensors state (the normalization stats).
- Saved ACT/Diffusion preprocessor steps: rename → to_batch → device → normalizer.
- Saved postprocessor steps: unnormalizer → device(cpu).
- SmolVLA adds the newline and tokenizer steps.

**Preprocessor input:** a dict of torch tensors, not numpy.
- `observation.state` as `(6,)` or `(B,6)`.
- Each camera as float32 CHW in [0,1], `(3,H,W)` or batched.
- Optionally `task` as `str` or `list[str]`; SmolVLA needs it.

It does not convert uint8 to float or HWC to CHW. It returns the batch normalized and on the target device.

**Policy and postprocessor outputs:** `select_action` returns a normalized `(B,dim)` tensor. The postprocessor returns unnormalized float32 `(B,dim)` on CPU, in dataset units.

**How `lerobot_eval.rollout` runs**
1. Calls `policy.reset()`, then `env.reset(seed)`.
2. Each step runs `preprocess_observation(obs)` from `envs/utils.py`. It expects gym-style keys:
   - `pixels`, a dict of camera → B×H×W×C uint8. It asserts uint8 and channel-last, divides by 255, converts to CHW and renames to `observation.images.<cam>`.
   - `agent_pos`, renamed to `observation.state`.
3. `add_envs_task` adds the task: from the env's `task_description` or `task` attribute, or `""` if neither exists.
4. Then `env_preprocessor` (identity except for LIBERO), the policy preprocessor, `select_action` under `inference_mode`, the postprocessor and `env_postprocessor`.
5. Finally `env.step(action.numpy())`.

**Minimal loop for your env (VERIFIED with an ACT checkpoint on CPU)**
```python
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
ckpt = f"{OUT}/checkpoints/last/pretrained_model"
cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.device = "cuda"   # optionally cfg.n_action_steps = 50
policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
                preprocessor_overrides={"device_processor": {"device": cfg.device}})
def to_batch(obs, task):
    b = {"observation.state": torch.from_numpy(obs["observation.state"]).float(), "task": task}
    for k in ("observation.images.front", "observation.images.wrist"):
        b[k] = torch.from_numpy(obs[k]).permute(2, 0, 1).float().div(255)
    return b
policy.reset(); obs = env.reset()                    # reset() at every episode start
for t in range(max_steps):
    with torch.inference_mode():
        a = policy.select_action(pre(to_batch(obs, task)))
    obs, done = env.step(post(a)[0].numpy())         # (6,) float32, dataset units
```
- The same loop works for Diffusion, which keeps its own observation history.
- For SmolVLA, the rename map is saved in the checkpoint's preprocessor, so the front/wrist keys are renamed automatically (from reading the code).
- `predict_action_chunk` returns `(1, chunk, 6)`.

## 6. Registering a custom env for evaluation in `lerobot-train`
**It is possible.** `--env.discover_packages_path=<pkg>` imports a package and its submodules before the CLI is parsed. That package would need to:
- Register a dataclass with `@EnvConfig.register_subclass("so101sim")`, defining `task`, `fps` and `features`, for example `agent_pos` (STATE (6,)), `pixels/front` (VISUAL (H,W,3)) and `ACTION`.
- Define `features_map` from those names to `observation.*` / `action`, and a `gym_kwargs` property.
- Register the gym id `f"{package_name}/{task}"`; `package_name` defaults to `gym_<type>`.

The env itself must:
- Return `{"pixels": {cam: HWC uint8}, "agent_pos": float32}`.
- Expose `_max_episode_steps` and either `task` or `task_description`.
- Report `is_success` through `final_info`.

`--eval.batch_size` must be no larger than `--eval.n_episodes` (both default to 50).

**My recommendation: not worth it for now.** Evaluation blocks training, only runs with `SyncVectorEnv`, and needs MuJoCo EGL rendering inside the training process. A standalone evaluation script using the loop in §5 on saved checkpoints is simpler. Add the plugin later only if you want `lerobot-eval` compatibility.

Everything is in `/tmp/claude-24441/-scratch-e1583535/cc20221c-31b8-4cc2-b268-3cd4267ef119/scratchpad/research/`:
- Test scripts in `scripts/`: `ds_roundtrip.py`, `infer_test.py`, `av_check.py`, `addframe_check.py`, `diff_smol_check.py`, `parse_check.py`.
- Test dataset: `out/ds_test`.
- pip dry-run report: `pip/report.json`.
- smolvla_base config files: `dl_smolvla_base/`.