# Prepare nuScenes data

Run commands from the repository root. The reference experiments read data through the following layout. A local `data/` directory or a symlink to an existing data tree both work.

```text
data/
  nuscenes/
    v1.0-trainval/
    samples/
    sweeps/
    maps/
  surroundocc/
    train_samples/
    val_samples/
  nuscenes_cam/
    nuscenes_infos_train_gaussian_ad_v4.pkl
    nuscenes_infos_val_gaussian_ad_v4.pkl
ckpts/
  r101_dcn_fcos3d_pretrain.pth
```

Obtain nuScenes v1.0 trainval, sensor data and map expansion files through the [nuScenes downloads](https://www.nuscenes.org/download). Prepare the occupancy labels following [SurroundOcc](https://github.com/weiyithu/SurroundOcc). The reference configurations read the v4 information files shown above. These contain scene-indexed frames and metadata used to align sensors, boxes, poses and occupancy labels. Paths embedded in the metadata must resolve on the machine running the experiment.

The repository includes tools for inspecting an existing PKL and converting generic nuScenes information files. Inspect both splits before launching a long job:

```bash
python tools/data/stats_gaussianad_pkl.py --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl
python tools/data/stats_gaussianad_pkl.py --pkl data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl
```

### Preparing alternative metadata

The [converter](../tools/data/convert_nuscenes_infos_to_gaussianad.py) reads generic `nuscenes_infos_train.pkl` and `nuscenes_infos_val.pkl` files, nuScenes records and map data. With the default paths above it writes **v6** files:

```bash
python tools/data/convert_nuscenes_infos_to_gaussianad.py --dataroot data/nuscenes
```

To experiment with v6, point both `train_dataset_config.imageset` and `val_dataset_config.imageset` at the generated files in a derived config. v6 reconstructs several fields, including velocity and future driving commands. Exact reproduction of the v4 experiments requires matching metadata. The v4 PKLs are external data files. The tracked converter generates v6 metadata.

### Checkpoints

Download the [R101 DCN FCOS3D backbone initialization](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) linked by the [upstream GaussianAD project](https://github.com/wzzheng/GaussianAD#train). Place it at `ckpts/r101_dcn_fcos3d_pretrain.pth` or set `load_from` to the corresponding local file in your config. Training writes full model checkpoints under the work directory. The repository does not bundle weights or a verified public download for the reported GaussPACE model.

Historical pseudo-label experiments also read optional depth and segmentation targets from `data/pseudo_labels/metric_3d_nusc/` and `data/pseudo_labels/grounded_sam_nusc/`. The occupancy and joint-planning recipes use the v4 occupancy metadata.
