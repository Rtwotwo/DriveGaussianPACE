# GaussPACE

**Physics-Constrained Gaussian Propagation and Completion for Future Occupancy Forecasting**

A vehicle moving forward brings new regions into view. Moving the Gaussians that describe the current scene can explain where existing objects go, but leaves little support for regions that were barely observed. GaussPACE forecasts occupancy with two sets of 3D Gaussians. One propagates the represented scene using motion states and a physically regularized rollout. The other predicts additional primitives from camera history at queries conditioned on future ego pose. The occupancy head combines both sets in the coordinate frame of each forecast.

This repository extends [GaussianAD](https://arxiv.org/abs/2412.10371) and also includes experiments that use forecast Gaussians to plan an ego trajectory.

## Forecasting on nuScenes

The occupancy experiment uses four frames from six cameras, current-frame box annotations for motion conditioning, and future ego poses for alignment and completion. These conditioning inputs matter when comparing the reported scores with a camera-only deployment setting. The model predicts a 120 × 120 × 8 occupancy grid at 0.5 m resolution and forecasts six steps at 0.5 s intervals.

The GaussPACE manuscript reports the following results on the nuScenes validation split with SurroundOcc labels. Values are percentages. The final column averages the **1, 2 and 3 s** endpoints.

| Metric | Current | 1 s | 2 s | 3 s | Future endpoint mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic mIoU | 27.20 | 23.42 | 20.27 | 17.38 | **20.36** |
| Occupied IoU | 39.94 | 36.56 | 33.07 | 29.31 | **32.98** |

The evaluator also prints an average over all **six** future steps. Use its individual 1, 2 and 3 s rows to compute the endpoint mean in the table.

## Run the occupancy experiment

1. Build the Python environment and CUDA operators using the [installation guide](docs/installation.md).
2. Prepare the [data, metadata and backbone initialization](docs/data.md).
3. From the repository root, train the [full-data V3-SE3 configuration](config/nuscenes_gs25600_v3_se3_full/nuscenes_gs25600_v3_se3_full.py) on one eight-GPU node:

```bash
OCC=config/nuscenes_gs25600_v3_se3_full
bash "$OCC/train.sh"
```

The reference schedule uses AdamW, a learning rate of `2e-4`, one sample per GPU and 20 epochs. Validation runs every four epochs. Outputs are written to `exp/nuscenes_gs25600_v3_se3_full/`. The trainer checks the work directory for `latest.pth` on startup. Set `WORK_DIR` to a new directory for an independent run.

To evaluate a trained occupancy checkpoint on one visible GPU:

```bash
OCC=config/nuscenes_gs25600_v3_se3_full
GPUS=0 CKPT=exp/nuscenes_gs25600_v3_se3_full/checkpoints/epoch_20.pth \
  bash "$OCC/test.sh"
```

The evaluator loads the checkpoint strictly and reports current and per-horizon semantic and occupied IoU. The R101 file used to initialize training contains backbone weights. Evaluation requires a trained full-model checkpoint.

## Planning with forecast Gaussians

The joint experiment lets the planner attend to a bank of Gaussians predicted from the observed history. One branch attends to each future step separately. A second branch attends across the horizon and proposes a correction to the trajectory. A learned gate scales this correction at each waypoint. The planning interface keeps the occupancy branch's privileged conditioning inputs out of the planner forward pass.

```bash
PLAN=config/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full
bash "$PLAN/train_joint_full.sh"
EPOCH=20 GPUS=0 bash "$PLAN/test_joint_full.sh"
```

This recipe trains the occupancy, map and planning components together from R101 initialization. For the frozen-frontend VAD comparison and older planner variants, see the [configuration guide](config/README.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| [config/](config/) | MMEngine experiment definitions, launchers and evaluators |
| [dataset/](dataset/) | nuScenes scene loading, transforms and planning metrics |
| [model/](model/) | Gaussian lifters, temporal encoders, occupancy heads, planners and native operators |
| [loss/](loss/) | Occupancy, motion, detection, map and trajectory objectives |
| [tools/data/](tools/data/) | Annotation conversion and inspection |
| [tools/viz/](tools/viz/) | Utilities for inspecting predictions |
| [train.py](train.py) | Training, validation and checkpoint management |

## Attribution and licensing

GaussPACE builds on GaussianAD and the OpenMMLab ecosystem. Experiments use [nuScenes](https://www.nuscenes.org/nuscenes) and [SurroundOcc](https://github.com/weiyithu/SurroundOcc) annotations. The repository carries an [Apache-2.0 license](LICENSE). Vendored components retain their own notices. In particular, the [local aggregation extension](model/head/localagg/setup.py) carries an Inria/GRAPHDECO non-commercial research and evaluation notice. Its header refers to a separate `LICENSE.md`. That document is missing from this source tree. The extension's redistribution terms still need to be confirmed before public release.
