"""Full-val planning-only evaluator for the isolated Stage-2 VAD model.

Unlike the repository-wide evaluator, this entry point never requests or
reports Future OCC.  Evaluation GT is used only after Planner inference to
compute the official v1/no-cumsum L2 and collision metrics.
"""

import argparse
import os
import os.path as osp
import sys
import time
import warnings

import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor


warnings.filterwarnings('ignore')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metric_v1_no_cumsum import compute_planner_metric_v1


def _silence_print(*args, **kwargs):
    del args, kwargs


def main(local_rank, args):
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    distributed = args.gpus > 1
    if distributed:
        address = os.environ.get('MASTER_ADDR', '127.0.0.1')
        port = os.environ.get('MASTER_PORT', '21876')
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{address}:{port}',
            world_size=args.gpus,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        cfg.gpu_ids = range(args.gpus)
        if local_rank != 0:
            import builtins
            builtins.print = _silence_print
    else:
        torch.cuda.set_device(0)

    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = args.log_name or f'test_plan_{timestamp}'
    logger = MMLogger(
        'selfocc', log_file=osp.join(args.work_dir, f'{log_name}.log'))
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(
        'Protocol: frozen V3-SE3-full epoch20 perception; original VADHead; '
        'predicted current Gaussian/agents/map only; v1 no-cumsum metric')
    logger.info(
        'GT scope: ego trajectory and future agent boxes are evaluator-only '
        'targets and never enter Planner inference')

    import model
    from dataset import get_dataloader
    from misc.metric_util import DetMetric

    network = build_segmentor(cfg.model)
    network.init_weights()
    if distributed and cfg.get('syncBN', True):
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = network.cuda()
    if distributed:
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        raw_model = network.module
    else:
        raw_model = network

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True,
    )

    if not osp.isfile(args.resume_from):
        raise FileNotFoundError(args.resume_from)
    checkpoint = torch.load(args.resume_from, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = raw_model.load_state_dict(state_dict, strict=True)
    logger.info(f'checkpoint: {args.resume_from}')
    logger.info(f'load_state_dict: {incompatible}')
    logger.info(f'validation samples: {len(val_loader.dataset)}')

    detector_metric = DetMetric(cfg, args.work_dir)
    planner_results = []
    metric_state = {'gt_num': 0}
    network.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            for key in list(data):
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()
            images = data.pop('img')
            result = network(
                imgs=images,
                metas=data,
                planner_eval_only=True,
            )

            command_mask = result['metas']['ego_fut_cmd'] == 1
            prediction = result['ego_fut_preds'][command_mask]
            planner_results.append(compute_planner_metric_v1(
                pred_ego_fut_trajs=prediction,
                gt_ego_fut_trajs=result['metas']['ego_fut_trajs'],
                gt_agent_boxes=result['metas']['gt_boxes_planner'][0],
                gt_agent_feats=result['metas']['attr_labels_planner'],
                fut_valid_flag=result['metas']['fut_valid_flag'],
            ))
            if index % cfg.print_freq == 0 and local_rank == 0:
                logger.info(f'[PLAN EVAL] Iter {index:5d}/{len(val_loader)}')

    gathered = detector_metric._after_epoch(
        local_rank,
        distributed,
        val_loader.dataset,
        {'planner_results': planner_results},
        metric_state,
    )
    if local_rank == 0:
        evaluation = val_loader.dataset.evaluate(
            gathered, metric='bbox', map_metric='chamfer', pipeline=None)
        logger.info(f'Planning evaluation: {evaluation}')

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-from', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-name', default='')
    arguments = parser.parse_args()
    arguments.gpus = torch.cuda.device_count()
    if arguments.gpus < 1:
        raise RuntimeError('no visible CUDA device')
    if arguments.gpus > 1:
        torch.multiprocessing.spawn(
            main, args=(arguments,), nprocs=arguments.gpus)
    else:
        main(0, arguments)
