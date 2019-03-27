"""
damagePrecision

plot the Precision-Recall, for a (N)random images

The precision-recall curve shows the tradeoff between precision
and recall for different threshold.

@author Adonis G & Tomas
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import damageDetection
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.gridspec as gridspec
import skimage.io

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
DAMAGE_DETECTION_DIR = os.path.join(ROOT_DIR, "logs")
DAMAGE_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/damage20190319T1825/mask_rcnn_damage_0020.h5")

config = damageDetection.DamageConfig()
DAMAGE_DIR = os.path.join(ROOT_DIR, "dataset/")


class InferenceConfig(config.__class__):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()


def get_ax(rows=1, cols=1, size=16):
  _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
  return ax


# Build validation dataset
dataset = damageDetection.DamageDataset()
dataset.load_custom(DAMAGE_DIR, "val")

dataset.prepare()
print("---------------------------------------------------------------------------")
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
print("---------------------------------------------------------------------------")


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=DAMAGE_DETECTION_DIR,
                              config=config)

weights_path = DAMAGE_WEIGHTS_PATH
# Or, uncomment to load the last model you trained
# weights_path = model.find_last()

# Load weights
print("---------------------------------------------------------------------------")
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("---------------------------------------------------------------------------")

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]


print("---------------------------------------------------------------------------")
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))
print("---------------------------------------------------------------------------")
results = model.detect([image], verbose=1)

r = results[0]

AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])

visualize.plot_precision_recall(AP, precisions, recalls)
plt.show()


# Display results
print("---------------------------------------------------------------------------")
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print("---------------------------------------------------------------------------")


# Draw precision-recall curve
image_ids = np.random.choice(dataset.image_ids, 7)
APs = []

for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])

    #AP = 1-AP
    print(r['rois'])
    APs.append(AP)
    print(APs)
AP = np.mean(APs)
print("------------")
print("mAP: ", np.mean(APs))
print("------------")

#image_ids = np.random.choice(dataset.image_ids, 5)

#image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

print("-----mask-------")
print (r['masks'])


visualize.plot_precision_recall(AP, precisions, recalls)
plt.show()





