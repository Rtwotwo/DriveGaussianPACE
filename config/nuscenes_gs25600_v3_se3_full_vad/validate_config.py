"""CPU-only structural validation for the isolated Stage-2 configuration."""

import argparse
import os
import sys

import torch
from mmengine import Config
from mmengine.utils import import_modules_from_strings
from mmseg.models import build_segmentor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(__file__), 'nuscenes_gs25600_v3_se3_full_vad.py')
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)

    assert cfg.max_epochs == 20
    assert cfg.train_dataset_config.num_samples == 0
    assert cfg.val_dataset_config.num_samples == 0
    assert cfg.train_loader.batch_size == 1
    assert cfg.val_loader.batch_size == 1
    assert cfg.model.type == 'BEVSegmentorV3SE3FullVADIsolated'
    assert cfg.model.planner_head.type == 'VADHead'
    assert len(cfg.loss.loss_cfgs) == 1
    assert cfg.loss.loss_cfgs[0].type == 'PlanLoss'

    network = build_segmentor(cfg.model)
    trainable = [
        (name, parameter.numel()) for name, parameter in network.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [name for name, _ in trainable
               if not name.startswith('planner_head.')]
    if invalid:
        raise AssertionError(f'non-Planner trainable parameters: {invalid[:8]}')
    count = sum(size for _, size in trainable)
    if count != 768164:
        raise AssertionError(
            f'expected 768164 original VADHead parameters, got {count}')
    frozen_modes = {
        name: getattr(network, name).training
        for name in network._FRONTEND_MODULES
    }
    if any(frozen_modes.values()):
        raise AssertionError(f'frontend module left in train mode: {frozen_modes}')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        source_state = checkpoint.get('state_dict', checkpoint)
        model_state = network.state_dict()
        unexpected = sorted(set(source_state) - set(model_state))
        missing = sorted(set(model_state) - set(source_state))
        mismatched = sorted(
            key for key in set(source_state) & set(model_state)
            if source_state[key].shape != model_state[key].shape)
        if unexpected:
            raise AssertionError(f'unexpected checkpoint keys: {unexpected[:8]}')
        if mismatched:
            raise AssertionError(f'shape mismatches: {mismatched[:8]}')
        invalid_missing = [
            key for key in missing if not key.startswith('planner_head.')
        ]
        if invalid_missing:
            raise AssertionError(
                f'non-Planner missing checkpoint keys: {invalid_missing[:8]}')
        print(
            f'checkpoint_compatibility=unexpected:0,shape_mismatch:0,'
            f'missing_planner_only:{len(missing)}')

    print('Stage-2 VAD config validation passed')
    print(f'load_from={cfg.load_from}')
    print('planner_inputs=' + ','.join(network._PLANNER_KEYS))
    print(f'trainable_parameters={count} (planner_head only)')
    print('per_gpu_batch=1; full_nuScenes=True; epochs=20')


if __name__ == '__main__':
    main()
