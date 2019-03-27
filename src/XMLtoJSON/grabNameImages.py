
"""
grabNamesImages

To grab names of the images in one folder,
and save in one document image.txt every image.

Next step you need to run convertXMLtoJson

@author Adonis & Tom
"""
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def grabNamesImages():
    path=[]
    train = os.path.join(ROOT_DIR, "train")
    val = os.path.join(ROOT_DIR, "val")
    path = [train,val]

    for file in path:
        files = os.listdir(file)
        print("---------")
        print(files)
        print("---------")
        for name in files:
            print(name)
        #        img_id = int(img_name.split('.')[0])
            imgs = []
            with open(file + '/image.txt', 'w') as f:  # /img/image.txt
                for item in files:
                    if (item.endswith('.jpg')):
                        # imgs.append(item)
                        # img2 = item.split('.jpg')[0]
                        # print (imgs)
                        # print (img2)
                        # f.write("%s\n" % img2)
                        f.write("%s\n" % item)

grabNamesImages()



