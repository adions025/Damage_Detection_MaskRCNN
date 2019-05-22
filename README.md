# Mask R-CNN for damage detection on Windmill Blade


## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
  ```
  $ git clone http://141.252.12.43/adions025/maskrcnn.git
   ```
2. Install dependencies
 ```
   pip install -r requirements.txt
 ```
3. Run setup from the repository root directory
 ```bash
    python setup.py install
 ``` 


# How to train in this project - Training Process

- you can find the class in src/main/**damageDetection.py**, allows to training your own dataset

```
$ python damageDetection.py train --dataset=/home/student_5/workspace/Mask_RCNN/dataset/ --weights=imagenet --logs=/home/student_5/workspace/Mask_RCNN/logs/
```



### if your annotations are made by labelimg you need to use this:
convert your annotations xml to json for diferences regions:
* you can also find this file inside of dataset folder, just run this file [ConverterXMLtoJson.py](https://github.com/adions025/XMLtoJson_Mask_RCNN) [moreinfo]
  
  ```
  $ python converterXMLtoJSON.py
   ```
   
 - before run put your images .jpg and your .xml file inside /train and /val
 - you need to have this structure :
 
  - /Mask_rcnn
    * /dataset
        * /train
        * /val
        * converterXMLtoJson.py
        

## Versioning
 * You can use binary segmentation version or multiclass, just use **version.sh** file
 * Make sure that this file has the necessary execution permissions.
     ```
     chmod +x versions.sh
     ```
 * You will get two folders with the different code versions in the previous path.
           
           
### Additional information:
* Mask R-CNN needs eggs, run you setup.py file to generate again.
* If you have installation problems, you can use the same enviroment (enviroment.yml) conda.
    ```
    $ conda env create -f environment.yml
    ```
* You can now activate the enviroment 
    ```
    $ cconda activate myenv
    ```
* You can find more info about how to manage conda enviroments [Creating an enviroment from an enviroment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) [moreinfo]

 
## Authors

* **Adonis González Godoy** - *object instance segmentation* - [NHL Stenden University](https://github.com/adions025)

   





