# Table of contents
1. Setup Environment
2. Test Environment
3. Projects

# Setup Environment
Follow [Installation](../INSTALL.md) Guide from Facebook.

## 1. Create a conda environment
```bash
# List all the installed envs
conda env list

# Create an env 
conda create --name dt2 python=3.7

# Get into the env 
conda activate dt2
# or the below cmd
source activate dt2

# Check env interpreter
which python

# List all packages in this env
source activate detectron2
pip list
```

## 2. Install requirements
```bash
# install dependencies: (use cu100 because colab is on CUDA 10.0)
pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
gcc --version
```

Install Jupyter notebook for remote working
```bash
pip install jupyter
pip install jupyterlab
```


## 3. Build Detectron2 from Source
```bash

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 
python -m pip install -e .
```

# Test Environment

Follow the Google Colab Tutorial [here](https://colab.research.google.com/drive/1u7YvZD8FjDX0xuHL8x5vUwkDfw4W5qry).

* Run the `run_inference.py` from the pretrained model:
```bash
cd nvq/exp/
ln -s ~/.torch/fvcore_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600 mask_r50.pkl
python run_inference.py
```

* Run `run_trainer.py` on `Balloon` dataset
```bash
python run_trainer.py
tensorboard --logdir output
```
* 
cd detectron2/datasets
# create a symbolic links to dataset
ln -s /home/quang/datasets/COCO_2017/coco coco

# 2 GPUs
python tools/train_net.py --num-gpus 2 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

python run_inference.py
```

### Where the pre-trained models of FAIR saved:

```markdown
https://github.com/facebookresearch/detectron2/issues/773

Yes, they are in ~/.torch/fvcore_cache/detectron2:
.
├── COCO-Detection
│   └── retinanet_R_101_FPN_3x
│   └── 138363263
│   ├── model_final_59f53c.pkl
│   └── model_final_59f53c.pkl.lock
├── COCO-InstanceSegmentation
│   ├── mask_rcnn_R_50_FPN_3x
│   │   └── 137849600
│   │   ├── model_final_f10217.pkl
│   │   └── model_final_f10217.pkl.lock
│   └── mask_rcnn_X_101_32x8d_FPN_3x
│   └── 139653917
│   ├── model_final_2d9806.pkl
│   └── model_final_2d9806.pkl.lock
├── COCO-PanopticSegmentation
│   └── panoptic_fpn_R_101_3x
│   └── 139514519
│   ├── model_final_cafdb1.pkl
│   └── model_final_cafdb1.pkl.lock
└── ImageNetPretrained
└── MSRA
├── R-50.pkl
└── R-50.pkl.lock
```
