"""
converterXMLtoJSON

To convert the annontation file in xml made by LabellimgTool in only one JSON file.
The polygon shape that had to be obtained from the rectangle shape.

Before to use this file you need to generate a image.txt, see description in grabNameImages.py

@author Adonis & Tom
"""

import xml.etree.cElementTree as ET
import cv2
import os
import json
import sys
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def grabNamesImages():
    path=[]
    train = os.path.join(ROOT_DIR, "train")
    val = os.path.join(ROOT_DIR, "val")
    path = [train,val]
    for file in path:
        files = os.listdir(file)
        for name in files:
            print(name)
            imgs = []
            with open(file + '/image.txt', 'w') as f:
                for item in files:
                    if (item.endswith('.jpg')):
                        f.write("%s\n" % item)
            f.close()
        print("save it in", file)


def XMLtoJson():
    path = []
    train = os.path.join(ROOT_DIR, "train")
    val = os.path.join(ROOT_DIR, "val")
    path = [train, val]

    for dir in path:
        imgs_list = open(dir+'/image.txt','r').readlines()
        print("-----------")
        print(imgs_list)
        print("-----------")
        images, bndbox, size, polygon = {}, {}, {}, {}
        # images = []

        count = 1
        total = len(imgs_list)

        all_json = {}
        # open("dataset.json", "w").close()
        var = {}
        for img in imgs_list:
            process_bar(count, total)
            count += 1
            if 'jpg' in img:
                img_name = img.strip().split('/')[-1]
                namexml = (img_name.split('.jpg')[0])
                print(namexml)

                images.update({"filename": img_name})

                xml_n = namexml + '.xml'

                tree = ET.ElementTree(file=dir+'/'+xml_n)
                root = tree.getroot()
                print(root)
                for child_of_root in root:
                    if child_of_root.tag == 'filename':
                        image_id = (child_of_root.text)
                        sizetmp = os.path.getsize(dir+'/'+image_id)
                    if child_of_root.tag == 'object':
                        for child_of_object in child_of_root:
                            if child_of_object.tag == 'name':
                                category_id = child_of_object.text
                            if child_of_object.tag == 'bndbox':
                                for child_of_root in child_of_object:
                                    if child_of_root.tag == 'xmin':
                                        xmin = int(child_of_root.text)
                                    if child_of_root.tag == 'xmax':
                                        xmax = int(child_of_root.text)
                                    if child_of_root.tag == 'ymin':
                                        ymin = int(child_of_root.text)
                                    if child_of_root.tag == 'ymax':
                                        ymax = int(child_of_root.text)

            xmintmp = int(xmax - xmin) / 2
            xvalue = int(xmin + xmintmp)

            ymintemp = int(ymax - ymin) / 2
            yvalue = int(ymin + ymintemp)

            regions = {}
            regionsTemp = ({"all_points_x": (xmin, xvalue, xmax, xmax, xmax, xvalue, xmin, xmin, xmin),
                            "all_points_y": (ymin, ymin, ymin, yvalue, ymax, ymax, ymax, yvalue, ymin)})

            damage = {"damage": "damage"}
            regions.update({"region_attributes": damage})

            shapes = {"shape_attributes": regionsTemp}
            regions.update(shapes)

            polygon.update({"name": "polygon"})
            regions.update(shapes)
            regions.update(polygon)

            regions.update(regions)
            regions1 = {"0": regions}
            regions = {"regions": regions1}
            images.update(regions)
            size = {"size": sizetmp}
            images.update(size)
            # print(images)

            all = {}
            print(all)

            all_json[img_name] = images.copy()

        with open(dir+'/'+"dataset.json", "a") as outfile:
            json.dump(all_json, outfile)

           # open(dir+'/'+"dataset.json", "w").close()



if __name__ == "__main__":

    grabNamesImages()
    ## load image list from txt file

    XMLtoJson()





