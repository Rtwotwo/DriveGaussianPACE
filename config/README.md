# Experiment configurations

Start with a full-data recipe below. Directory names also identify historical runs, so the tasks, inputs and checkpoint requirements are listed explicitly.

| Experiment | Inputs and training objective | Entry point |
| --- | --- | --- |
| V3-SE3 occupancy | Four-frame camera history, current boxes and future poses. Trains current/future occupancy, detection and map components over the full train split. | [Training and occupancy evaluation](nuscenes_gs25600_v3_se3_full/) |
| Joint future-Gaussian planning | Trains the V3-SE3 frontend and a gated, per-frame/global residual planner together. The planner consumes predicted future scene features. | [Joint training and evaluation](nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/) |
| Frozen frontend with a VAD planner | Extracts a compatible perception frontend from a trained checkpoint and trains a planning backend. Check the source checkpoint and evaluator before comparing results. | [Stage-two configuration](nuscenes_gs25600_v3_se3_epoch20_ft_plan_vad_full/) |
| Frozen occupancy with truncated DDIM planning | Experimental diffusion-based planner. The experiment guide documents its initialization and sampling settings. | [Experiment guide](nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim/README.md) |

MMEngine resolves each Python config through its `_base_` chain. Earlier `gtbox_oracle` configurations include box-conditioned experiments, and some use subsampled splits. When comparing runs, record the resolved config, checkpoint epoch, input conditioning, split and evaluator. The full-data occupancy script reports six future time steps; the manuscript summarizes the endpoints at 1, 2 and 3 seconds.
