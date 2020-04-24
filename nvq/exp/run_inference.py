# python run_ference.py

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger(output=None)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# Load image
import os
os.system("wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg")
im = cv2.imread("input.jpg")
# plt.imshow(im)
# plt.show();

# create config and predictor
cfg = get_cfg()

# add project-specific config (e.g., TensorMask) 
# here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  

# Find a model from detectron2's model zoo.
# You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. 
# See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format 
# for specification
outputs["instances"].pred_classes
outputs["instances"].pred_boxes

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite('input_mask.jpg', v.get_image())

# plt.imshow(v.get_image()[:, :, ::-1])
# plt.show();
