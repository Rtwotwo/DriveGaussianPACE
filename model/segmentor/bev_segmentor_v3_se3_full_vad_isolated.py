"""Frozen V3-SE3 full-data perception followed by the original VAD planner.

This is an isolated Stage-2 model.  The prediction path exposed to VADHead is
strictly limited to predicted current-frame agents, map vectors and the
current 28-D Gaussian bank.  Future Gaussians, motion offsets, occupancy
labels and all planning ground truth are excluded from the Planner forward.
"""

import torch
from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class BEVSegmentorV3SE3FullVADIsolated(BEVSegmentor):
    """Train an original VADHead on a completely frozen V3-SE3 frontend."""

    _FRONTEND_MODULES = (
        'img_backbone',
        'img_neck',
        'lifter',
        'encoder',
        'temporal_encoder',
        'decoder',
        'map_decoder',
        'head',
    )
    _PLANNER_KEYS = (
        'final_box_dicts',
        'all_cls_scores',
        'all_pts_preds',
        'gaussian_output',
    )
    # These fields may be retained by the dataloader for supervision and
    # metrics, but are never shown to the frozen prediction frontend.
    _PREDICTION_META_DENYLIST = frozenset((
        'attr_labels_planner',
        'ego_fut_cmd',
        'ego_fut_masks',
        'ego_fut_trajs',
        'flow_info',
        'fut_valid_flag',
        'future_lidar2global',
        'occ_cam_mask',
        'occ_label',
    ))

    def __init__(self, *args, planner_head=None, **kwargs):
        planner_type = (planner_head.get('type')
                        if hasattr(planner_head, 'get') else None)
        if planner_type != 'VADHead':
            raise ValueError(
                'BEVSegmentorV3SE3FullVADIsolated requires the original '
                f'VADHead, got {planner_type!r}')
        super().__init__(*args, planner_head=planner_head, **kwargs)
        self._freeze_frontend()

    def _freeze_frontend(self):
        """Freeze parameters and stateful behaviour of every perception block."""
        for name in self._FRONTEND_MODULES:
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        """Keep the frontend in eval mode when the runner enters train mode."""
        super().train(mode)
        self._freeze_frontend()
        self.planner_head.train(mode)
        return self

    @staticmethod
    def _detach_tree(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {
                key: BEVSegmentorV3SE3FullVADIsolated._detach_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                BEVSegmentorV3SE3FullVADIsolated._detach_tree(item)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                BEVSegmentorV3SE3FullVADIsolated._detach_tree(item)
                for item in value
            )
        return value

    @classmethod
    def _prediction_metas(cls, metas):
        """Remove supervision/oracle fields before frozen prediction forward."""
        if not isinstance(metas, dict):
            raise TypeError(f'metas must be a dict, got {type(metas)!r}')
        return {
            key: value for key, value in metas.items()
            if (not key.startswith('gt_')
                and key not in cls._PREDICTION_META_DENYLIST)
        }

    def _forward_frozen_frontend(self, imgs, metas, points, kwargs):
        """Run current-scene perception without autograd or GT motion input."""
        results = {
            'imgs': imgs,
            'metas': self._prediction_metas(metas),
            'points': points,
        }
        results.update(kwargs)
        with torch.no_grad():
            results.update(self.extract_img_feat(**results))
            results.update(self.lifter(**results))
            results.update(self.encoder(**results))
            if hasattr(self, 'temporal_encoder'):
                # No ``gt_boxes`` key is present.  The current Gaussian is
                # produced normally; only the unused oracle-motion offset is
                # disabled.
                results.update(self.temporal_encoder(**results))
            results.update(self.decoder(results))
            if hasattr(self, 'map_decoder'):
                # MapTRv2.forward() defaults to ``return_loss=True`` even in
                # eval mode and therefore unconditionally reads map GT.  The
                # prediction part of MapTRv2.forward_train() is exactly this
                # head call; use it directly so GT never enters the Stage-2
                # prediction graph.
                map_head = getattr(self.map_decoder, 'pts_bbox_head', None)
                if map_head is None:
                    raise TypeError(
                        'GT-free Stage-2 requires map_decoder.pts_bbox_head')
                if 'encoded_spconv_tensor' not in results:
                    raise KeyError(
                        'missing encoded_spconv_tensor for GT-free map decode')
                map_outputs = map_head(results['encoded_spconv_tensor'])
                required_map_keys = {'all_cls_scores', 'all_pts_preds'}
                if (not isinstance(map_outputs, dict)
                        or not required_map_keys.issubset(map_outputs)):
                    raise KeyError(
                        'GT-free map head did not return all_cls_scores and '
                        'all_pts_preds')
                results.update(map_outputs)
        return results

    @torch.no_grad()
    def _render_current_occ(self, representation_temp, metas):
        """Render current OCC only; never call the Future/Oracle flow branch."""
        occ_xyz = metas['occ_xyz'].to(self.head.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.head.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.head.zero_tensor.device)
        sampled_xyz, sampled_label = self.head._sampling(
            occ_xyz, occ_label, None)
        gaussians = representation_temp['gaussian']
        means, original_opacity, opacities, scales, cov_inv = (
            self.head.prepare_gaussian_args(gaussians))
        batch_size, gaussian_count = means.shape[:2]
        semantics = self.head.aggregator(
            sampled_xyz.clone().float(),
            means,
            original_opacity.reshape(batch_size, gaussian_count),
            opacities,
            scales,
            cov_inv,
        )[None].transpose(1, 2)
        return {
            'pred_occ': [semantics],
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
        }

    def forward(self, imgs=None, metas=None, points=None,
                extra_backbone=False, occ_only=False, rep_only=False,
                planner_eval_only=False, **kwargs):
        if extra_backbone:
            with torch.no_grad():
                return self.forward_extra_img_backbone(imgs=imgs)

        results = self._forward_frozen_frontend(
            imgs=imgs, metas=metas, points=points, kwargs=kwargs)

        missing = [key for key in self._PLANNER_KEYS if key not in results]
        if missing:
            raise KeyError(f'missing frozen VAD input predictions: {missing}')
        current_gaussian = results['gaussian_output']
        if not torch.is_tensor(current_gaussian):
            raise TypeError('gaussian_output must be a tensor')
        if current_gaussian.ndim != 3 or current_gaussian.shape[-1] != 28:
            raise ValueError(
                'expected current gaussian_output [B,G,28], got '
                f'{tuple(current_gaussian.shape)}')
        if current_gaussian.shape[0] != 1:
            raise AssertionError(
                'the original VADHead requires per-GPU batch_size=1')

        planner_inputs = {
            key: self._detach_tree(results[key])
            for key in self._PLANNER_KEYS
        }
        if tuple(planner_inputs) != self._PLANNER_KEYS:
            raise AssertionError('unexpected field reached VADHead')
        planner_outputs = self.planner_head(planner_inputs)

        # Return only fields required by PlanLoss/evaluation so large frozen
        # feature tensors can be released before backward.
        outputs = dict(planner_outputs)
        outputs['all_cls_scores'] = results['all_cls_scores']
        outputs['all_pts_preds'] = results['all_pts_preds']
        outputs['metas'] = metas
        outputs['ego_fut_trajs'] = metas['ego_fut_trajs']

        # train.py requires pred_occ during its periodic validation.  Render
        # only current OCC after Planner prediction; it cannot affect L2/CR.
        # The dedicated external Planner evaluator sets planner_eval_only=True
        # and skips even this frozen diagnostic render.
        if not self.training and not planner_eval_only:
            outputs.update(self._render_current_occ(
                results['representation_temp'], metas))
        return outputs
