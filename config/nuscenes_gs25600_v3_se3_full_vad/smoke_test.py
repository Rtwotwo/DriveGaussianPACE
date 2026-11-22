"""One-GPU real-batch validation for the isolated current-Gaussian VAD model.

The script performs no optimizer step and writes no checkpoint.  It checks a
real frontend/MapTR/VAD forward, one PlanLoss backward, the frozen-gradient
boundary, and invariance of Planner predictions to oracle/GT metadata.
"""

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


def _cuda_tensors(data):
    for key in list(data):
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda(non_blocking=False)
    return data


def _perturb_supervision(metas, model):
    perturbed = dict(metas)
    denied = model._PREDICTION_META_DENYLIST
    for key, value in list(perturbed.items()):
        if key.startswith('gt_') or key in denied:
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    perturbed[key] = torch.randn_like(value)
                else:
                    perturbed[key] = torch.zeros_like(value)
            elif key not in ('ego_fut_trajs',):
                perturbed[key] = None
    # The wrapper appends this target after Planner inference, so retain a
    # tensor with the correct shape while changing every value.
    if isinstance(metas.get('ego_fut_trajs'), torch.Tensor):
        perturbed['ego_fut_trajs'] = torch.randn_like(metas['ego_fut_trajs'])
    return perturbed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--py-config',
        default='config/nuscenes_gs25600_v3_se3_full_vad/'
                'nuscenes_gs25600_v3_se3_full_vad.py')
    parser.add_argument(
        '--checkpoint',
        default='exp/nuscenes_gs25600_v3_se3_full_vad/bootstrap/'
                'v3_se3_full_epoch20_weights.pth')
    parser.add_argument('--forward-only', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('the real-batch smoke test requires one CUDA GPU')
    cfg = Config.fromfile(args.py_config)
    import_modules_from_strings(**cfg.custom_imports)
    # Avoid spawning data workers for a one-batch diagnostic.
    cfg.train_loader.num_workers = 0
    cfg.val_loader.num_workers = 0

    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    network = build_segmentor(cfg.model)
    network.init_weights()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    incompatible = network.load_state_dict(
        checkpoint.get('state_dict', checkpoint), strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith('planner_head.')
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise AssertionError(
            f'bad checkpoint boundary: {incompatible}, '
            f'non_planner_missing={invalid_missing[:8]}')
    network = network.cuda()

    train_loader, _ = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
    )
    batch = _cuda_tensors(next(iter(train_loader)))
    images = batch.pop('img')

    network.eval()
    with torch.no_grad():
        first = network(
            imgs=images, metas=batch, planner_eval_only=True)
        altered_metas = _perturb_supervision(batch, network)
        second = network(
            imgs=images, metas=altered_metas, planner_eval_only=True)
    difference = (first['ego_fut_preds'] - second['ego_fut_preds']).abs().max()
    if difference.item() != 0.0:
        raise AssertionError(
            f'Planner prediction changed after GT perturbation: {difference}')
    print('real_batch_forward=passed')
    print(f'ego_fut_preds_shape={tuple(first["ego_fut_preds"].shape)}')
    print(f'gt_perturbation_max_abs_diff={difference.item():.8f}')

    if not args.forward_only:
        network.train()
        result = network(imgs=images, metas=batch, current_epoch=0)
        loss_inputs = {'metas': batch}
        for target, source in cfg.loss_input_convertion.items():
            loss_inputs[target] = (
                result[source] if target in result
                else result['metas'].get(source))
        loss_function = OPENOCC_LOSS.build(cfg.loss).cuda()
        loss, _ = loss_function(loss_inputs)
        loss.backward()

        frontend_grads = [
            name for name, parameter in network.named_parameters()
            if (not name.startswith('planner_head.')
                and parameter.grad is not None)
        ]
        planner_grads = [
            name for name, parameter in network.named_parameters()
            if name.startswith('planner_head.') and parameter.grad is not None
        ]
        if frontend_grads:
            raise AssertionError(
                f'frozen frontend received gradients: {frontend_grads[:8]}')
        if not planner_grads:
            raise AssertionError('Planner received no gradients')
        print(f'plan_loss_backward=passed, loss={loss.item():.6f}')
        print(f'planner_tensors_with_grad={len(planner_grads)}')
        print('frontend_tensors_with_grad=0')

    print(
        'peak_cuda_memory_gib=%.2f' %
        (torch.cuda.max_memory_allocated() / (1024 ** 3)))


if __name__ == '__main__':
    main()
