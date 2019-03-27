"""
damateTesting
Allows testing diferents models made by damageDetection Class

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
from mrcnn import utils
import damageDetection

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #only for avoid a warning in tensorflow

tf.summary

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#DAMAGE_DETECTION_MODEL = os.path.join(ROOT_DIR, "logs/damage20190319T1825/mask_rcnn_damage_0020.h5") #this model was trainng 1000 step per epoch * 20 epocchs
DAMAGE_DETECTION_MODEL = os.path.join(ROOT_DIR, "logs/damage20190326T1616/mask_rcnn_damage_0020.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(damageDetection.DamageConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
model.load_weights(DAMAGE_DETECTION_MODEL, by_name=True)

print("Loading weights ", DAMAGE_DETECTION_MODEL)

class_names = ['BG', 'damage']


#path = os.path.join(ROOT_DIR, 'dataset/val/IMG_20180413_103507.jpg')
#path = os.path.join(ROOT_DIR, 'dataset/val/*.jpg')



val = os.path.join(ROOT_DIR, "dataset/val")
resultados = os.path.join(ROOT_DIR, "dataset/results")

files = os.listdir(val)
with open(val + '/imageval.txt', 'w') as f:  # /img/image.txt
    for item in files:
        if (item.endswith('.jpg')):
            f.write("%s\n" % item)

imgs_list = open(val+'/imageval.txt','r').readlines()
for img in imgs_list:
    if 'jpg' in img:
        img_name = img.strip().split('/')[-1]

        image = skimage.io.imread(val+'/'+img_name)
        results = model.detect([image], verbose=0)
        r = results[0]

        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                           class_names, r['scores'])

        #skimage.io.imsave(resultados + '/' + img_name, save)

        visualize.save_image(image, img_name, r['rois'], r['masks'],r['class_ids'],r['scores'],class_names,mode=0)
        visualize.di


#def save_image(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
 #             scores_thresh=0.1, save_dir=None, mode=0):

'''
visualize.display_weight_stats(model)


# Pick layer types to display
LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES,
                layers))
# Display Histograms
fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)),
                       gridspec_kw={"hspace":1})


for l, layer in enumerate(layers):
    weights = layer.get_weights()
    for w, weight in enumerate(weights):
        tensor = layer.weights[w]
        ax[l, w].set_title(tensor.name)
        _ = ax[l, w].hist(weight[w].flatten(), 50)

'''
print("estamo bien")


print('----------------------')
print(class_names)
print('----------------------')

print('----------------------')
for i in r:
    print (r[i])
print('----------------------')

print('----------------------')
print('im here')
print('----------------------')





#################################
####      para usar camara   ####
#################################


'''
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while True:
    ret, frame = capture.read()
    results = model.detect([frame], verbose=0)
    r = results[0]

    frame = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

    cv2.imshow('frame')
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

'''
