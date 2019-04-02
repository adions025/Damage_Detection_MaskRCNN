# Mask R-CNN for damage detection on Windmill Blade


## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
  ```
  $ git clone https://adonisgonzalez@bitbucket.org/aicomputervision/damagedetectionwindmill.git
   ```
2. Install dependencies
 ```
   pip install -r requirements.txt
 ```
3. Run setup from the repository root directory
 ```bash
    python3 setup.py install
 ``` 


# How to train in this project - Training Process

- you can find the class in src/main/**damageDetection.py**, allows to training your own dataset
```
python samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco
```
```
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet
```



### if your annotations are made by labelimg you need to use this:
convert your annotations xml to json for diferences regions:
* you can also find this file inside of dataset folder, just run this file
###[ConverterXMLtoJson.py](https://github.com/adions025/XMLtoJson_Mask_RCNN)
  ```
  $ python converterXMLtoJSON.py
   ```
 * before run put your images .jpg and your .xml file inside /train and /val
 * you need to have this structure :
    * Mask_rcnn
        * dataset
            * train
            * val
 * after run you will have one file dataset.json in /val and /train
           
     
   





