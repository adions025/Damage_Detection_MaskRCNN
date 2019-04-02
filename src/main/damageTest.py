import skimage

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import mrcnn.model as modellib
matplotlib.use('tkagg')
import tensorflow as tf
import os
import random
import sys
import damageDetection
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import re
import time
import numpy as np


DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
DAMAGE_DETECTION_DIR = os.path.join(ROOT_DIR, "logs")
DAMAGE_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/damage20190402T1016/mask_rcnn_damage_0020.h5")
val = os.path.join(ROOT_DIR, "dataset/val")

config = damageDetection.DamageConfig()
DAMAGE_DIR = os.path.join(ROOT_DIR, "dataset/")

class InferenceConfig(config.__class__):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


# Load validation dataset
dataset = damageDetection.DamageDataset()
dataset.load_custom(DAMAGE_DIR, "val")

# Must call before using the dataset
dataset.prepare()


with tf.device(DEVICE):
  model = modellib.MaskRCNN(mode="inference", model_dir=DAMAGE_DETECTION_DIR,
                            config=config)
print("Loading weights ", DAMAGE_WEIGHTS_PATH)
model.load_weights(DAMAGE_WEIGHTS_PATH, by_name=True)


class_names = ['BG','Erosion 1','Erosion 2','Erosion 3','SD','SD 1', 'SD 2', 'SD 3','B&C','B&C 1','B&C 2','B&C 3', 'B&C 4','Dirt']

path = os.path.join(ROOT_DIR, "dataset/val/IMG_20180413_094756.jpg")


image = skimage.io.imread(path)

# Run detection
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])