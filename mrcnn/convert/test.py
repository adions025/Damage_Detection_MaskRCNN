import os
import sys
import json
import os
import skimage.draw
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath("/Mask_RCNN")


dataset_dir = os.path.dirname(os.path.realpath(__file__))

#assert subset in ["train", "val"]
#dataset_dir = os.path.join(dataset_dir, subset)

annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))


from xml.dom import minidom

# parse an xml file by name
mydoc = minidom.parse('items.xml')

items = mydoc.getElementsByTagName('item')
# print(annotations1)
annotations = list(annotations1.values())  # don't need the dict keys

print('----------------------------------')
#print(annotations)
print('----------------------------------')


def add_image(self, source, image_id, path, **kwargs):
    image_info = {
        "id": image_id,
        "source": source,
        "path": path,
    }
    image_info.update(kwargs)


for a in annotations:

    polygons = [r['shape_attributes'] for r in a['regions'].values() ]


    for r in a['regions'].values():
        print(r)
    for r in a['shape_attributes'].values():
        print(r)


    print('----------------------------------')
    print (polygons)
    print('----------------------------------')

    print('----------------------------------')

    print('----------------------------------')

    image_path = os.path.join(dataset_dir, a['filename'])
    print(a['filename'])
    image = skimage.io.imread(image_path)

    height, width = image.shape[:2]
    print('----------------------------------')
    print(image)
    print('----------------------------------')

    print('----------------------------------')
    print(height)
    print('----------------------------------')

    print('----------------------------------')
    print(width)
    print('----------------------------------')


    add_image("damage",dataset_dir, image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)

    break





