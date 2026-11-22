"""Stage-2: frozen V3-SE3-full epoch20 perception + original VADHead.

The Planner consumes predicted current-frame agents, map vectors and the
current Gaussian bank only.  No offset, Future Gaussian, OCC logits or GT
field is part of the Planner forward.  Only ``planner_head`` is optimized.
"""

from copy import deepcopy
import os


_base_ = [
    '../nuscenes_gs25600_v3_se3_full/nuscenes_gs25600_v3_se3_full.py'
]

custom_imports = deepcopy(_base_.custom_imports)
custom_imports['imports'] = list(custom_imports['imports']) + [
    'model.segmentor.bev_segmentor_v3_se3_full_vad_isolated',
]


# Full nuScenes Stage-2 schedule.  Per-rank batch must stay at one because the
# original GaussianAD VADHead indexes only the first sample of each rank.
max_epochs = int(os.environ.get('MAX_EPOCHS', 20))
eval_every_epochs = int(os.environ.get('EVAL_EVERY_EPOCHS', 4))
train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)
train_loader = dict(batch_size=1)
val_loader = dict(batch_size=1)


# The launch script normally points this at a weights-only copy made from this
# exact checkpoint.  Direct Config use still loads the original checkpoint.
load_from = os.environ.get(
    'FRONTEND_CKPT',
    'exp/nuscenes_gs25600_v3_se3_full/checkpoints/epoch_20.pth',
)

# At 8 GPUs: per-GPU batch=1, global batch=8, LR=2e-4.  train.sh scales LR
# linearly for a smaller GPU count while keeping the immutable per-rank batch.
lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(),
)
grad_max_norm = 35
warmup_iters = 500

frozen_modules = [
    'img_backbone',
    'img_neck',
    'lifter',
    'encoder',
    'temporal_encoder',
    'decoder',
    'map_decoder',
    'head',
]
find_unused_parameters = False
static_graph = False


_attention = dict(
    type='MultiheadAttention',
    embed_dims=128,
    num_heads=8,
    dropout=0.1,
)
_vad_decoder = dict(
    type='CustomTransformerDecoder',
    num_layers=1,
    return_intermediate=False,
    transformerlayers=dict(
        type='MyCustomBaseTransformerLayer',
        attn_cfgs=[_attention],
        feedforward_channels=256,
        ffn_dropout=0.1,
        batch_first=False,
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
    ),
)

model = dict(
    type='BEVSegmentorV3SE3FullVADIsolated',
    history_no_grad=False,
    # Kept only as a frozen current-OCC diagnostic renderer during the
    # training runner's periodic validation.  Its Future branch is not called.
    head=dict(type='GaussianHeadFrontierV3Isolated'),
    planner_head=dict(
        _delete_=True,
        type='VADHead',
        embed_dims=128,
        fut_ts=6,
        fut_mode=6,
        ego_fut_mode=3,
        ego_agent_decoder=deepcopy(_vad_decoder),
        ego_map_decoder=deepcopy(_vad_decoder),
        ego_gaussian_decoder=deepcopy(_vad_decoder),
    ),
)


# Exact original GaussianAD Planner objective: trajectory L1 plus predicted-map
# boundary and direction terms.  col_loss_weight keeps its default 0.0, so no
# new collision loss or Future-Gaussian auxiliary supervision is introduced.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[dict(type='PlanLoss', weight=10.0)],
)
loss_input_convertion = dict(
    _delete_=True,
    ego_fut_preds='ego_fut_preds',
    ego_fut_gt='ego_fut_trajs',
    ego_fut_masks='ego_fut_masks',
    ego_fut_cmd='ego_fut_cmd',
    all_cls_scores='all_cls_scores',
    all_pts_preds='all_pts_preds',
)
