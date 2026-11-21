# Installation

Use a Linux machine with an NVIDIA GPU, a CUDA-capable driver, the CUDA 11.8 toolkit (including `nvcc`), a compatible C++ compiler and Conda. Run the commands from the repository root. The reference environment uses Python 3.8, PyTorch 2.0.0 with CUDA 11.8, and MMCV 2.0.1.

## Python environment

```bash
conda create -n gaussianad python=3.8.16 -y
conda activate gaussianad
python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install openmim
python -m pip install mmengine==0.10.7
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
python -m pip install numpy==1.24.4 spconv-cu117==2.3.6 timm==1.0.26 gsplat==1.5.3 nuscenes-devkit==1.1.11 Shapely==1.8.5.post1
python -m pip install ninja SharedArray easydict tensorboard tensorboardX pyquaternion "matplotlib<3.6"
python -m pip install git+https://github.com/NVIDIA/gpu_affinity
```

## CUDA extensions

Set `CUDA_HOME` to your toolkit location if the compiler cannot find CUDA. Build the deformable aggregation, local aggregation and point-cloud operators against the PyTorch installation above. Each command below starts from the repository root.

```bash
(cd model/encoder/gaussian_encoder/ops && python setup.py build_ext --inplace)
(cd model/head/localagg && python -m pip install --no-build-isolation -e .)
python model/ops/setup.py develop
```

Check the environment with the same interpreter that will launch training:

```bash
python -c "import torch, mmcv; print(torch.__version__, torch.version.cuda, mmcv.__version__, torch.cuda.device_count())"
python -c "import torch; from local_aggregate import _C; from model.ops.iou3d_nms import iou3d_nms_cuda; from model.encoder.gaussian_encoder.ops import deformable_aggregation_ext; print(_C.__file__, iou3d_nms_cuda.__file__, deformable_aggregation_ext.__file__)"
python train.py --help
```

The import check prints all three shared-library paths. In a reused environment, check that they resolve to the extensions you just built.

The launchers use `python` and `torchrun` from the active environment. Set `PYTHON` or `TORCHRUN` to explicit executable paths when several environments share a machine. For 3D visualization, install Mayavi, PyQt5 and PyVirtualDisplay. A headless renderer also requires a working display backend such as Xvfb.
