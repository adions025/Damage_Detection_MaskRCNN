"""
damateTesting
Allow predict and save this images with predictions in /results folder

How to use?
1) Make sure you have a folder called results.
                Mask_rcnn
                    |-- dataset
                        |-- results
                        |-- train
                        `-- val
2)You have your images in dataset/val



@author: Adonis & Tom
"""

import sys
import skimage.io
import matplotlib
import matplotlib
matplotlib.use('tkagg')
import tensorflow as tf
import os
from mrcnn import visualize
import mrcnn.model as modellib
import damageDetection
import matplotlib as mpl
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #only for avoid a warning in tensorflow


#PATHS
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#DAMAGE_DETECTION_MODEL = os.path.join(ROOT_DIR, "logs/damage20190329T1140/mask_rcnn_damage_0100.h5")#workfistsprint
DAMAGE_DETECTION_MODEL = os.path.join(ROOT_DIR, "logs/damage20190403T1717/mask_rcnn_damage_0050.h5")
val = os.path.join(ROOT_DIR, "dataset/val")
resultados = os.path.join(ROOT_DIR, "dataset/results")


#MODEL
class InferenceConfig(damageDetection.DamageConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(DAMAGE_DETECTION_MODEL, by_name=True)

class_names = ['BG','Erosion','SD','B&C', 'Dirt']

print("--------------------------------------------")
print("Loading weights ", DAMAGE_DETECTION_MODEL)
print("--------------------------------------------")


#mpl.rcParams["savefig.directory"] = resultados #it doesnt work

'''
just create a new imageval.txt file, with new validation dataset
if you use the same validation dataset, you can delete this loop
'''
files = os.listdir(val)
with open(val + '/imageval.txt', 'w') as f:  # /img/image.txt
    for item in files:
        if (item.endswith('.jpg')):
            f.write("%s\n" % item)

#read the new dataset and copy in results

imgs_list = open(val+'/imageval.txt','r').readlines()
''''
for img in imgs_list:
    if 'jpg' in img:
        img_name = img.strip().split('/')[-1]
        os.system('cp'+ ' '+ val+'/'+img_name + ' '+resultados)
'''
#save the images in /results
for img in imgs_list:
    if 'jpg' in img:
        img_name = img.strip().split('/')[-1]
        image = skimage.io.imread(val+'/'+img_name)
        results = model.detect([image], verbose=0)
        r = results[0]
        path = resultados+'/'+img_name
        visualize.save_images(path,image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        #visualize.save_image(image, img_name, r['rois'], r['masks'],
            #r['class_ids'],r['scores'],class_names,scores_thresh=0.9,mode=0)

print("--------------------------------------------")
print("-------Downloading predictions finish-------")
print("--------------------------------------------")



