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
DAMAGE_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/damage20190403T1717/mask_rcnn_damage_0050.h5")
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

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

#class_names = ['BG','Erosion','SD', 'B&C','Dirt']
path = os.path.join(ROOT_DIR, "dataset/val/DSC_9535.jpg")
image = skimage.io.imread(path)

# Run detection
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'])