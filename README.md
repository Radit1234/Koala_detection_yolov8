# Koala_detection_yolov8
Description : Object detection is a major area of computer vision applications. For the task I had a set of images that contain instances of Koala. My main goal was  to preprocess the images and build a dataset, train an object detection model using the prepared dataset, and develop an API for getting inferences from the model.
## Data Annotation

 As it is a object detection project so we have to annotate those image so that our model can understand how the real Koala looks like. So for this parpose I used the www.cvat.ai .Using this website I markdown my object with leveling and later downloaded the files.   
## Dataset Structure 

Then I change the project export format to YOLO 1.1 and then downloaded the level data file. The directory has same fie name as main images data but the file format are different . Here are all in text format . Each one of this text row is koala .Here, first character is class, 4 numbers are Bounding box. First number is center of the Bounding box and then height and width of the bounding box are there. We need this level numbers for training the Dataset . So, levels are in YOLO format .  I also created a YAML file to direct the path of the data .
## Train Yolov8

Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

We hope that the resources here will help you get the most out of YOLOv8. Please browse the YOLOv8 Docs for details, raise an issue on GitHub for support, and join our Discord community for questions and discussions!

To request an Enterprise License please complete the form at Ultralytics Licensing.



Ultralytics GitHub  Ultralytics LinkedIn  Ultralytics Twitter  Ultralytics YouTube  Ultralytics TikTok  Ultralytics Instagram  Ultralytics Discord
Documentation
See below for a quickstart installation and usage example, and see the YOLOv8 Docs for full documentation on training, validation, prediction and deployment.

Install
Pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

PyPI version Downloads

pip install ultralytics
For alternative installation methods including Conda, Docker, and Git, please refer to the Quickstart Guide.

Usage
CLI
YOLOv8 may be used directly in the Command Line Interface (CLI) with a yolo command:

yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
yolo can be used for a variety of tasks and modes and accepts additional arguments, i.e. imgsz=640.

## Results

Here are the results:
![train_batch0](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/d16698cb-878c-4c04-ae16-133f930838b0)
![train_batch1](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/709d30f3-23f2-4a09-b716-568a83f8b79f)
![train_batch2](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/0d722df9-836c-4a64-b57c-aa76c631e9d9)


## Results graphs

[PR_curve](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/a5abfb1f-dcb8-44a6-b322-6fd1ab80c2b1)

![R_curve](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/f65a7640-5d28-4777-820f-36164943b8e7)

![F1_curve](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/c1b22037-23df-4bf5-99e0-b4cb95a61e0c)

![labels](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/9785b3ad-f6a0-4820-8b68-ce22bc83333a)

![labels_correlogram](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/48bccc9f-bc8f-42e1-9c7a-f2bcb4491fc5)

![P_curve](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/01386e0a-8a1e-413b-9a7f-e4be2b689165)


![results](https://github.com/Radit1234/Koala_detection_yolov8/assets/48798988/b1024091-703a-4e48-b4a8-bfc3bb319971)

## API Reference

#### Get all items

1. Import all necessary libraries
In this tutorial, we will use :

“flask” package: from which we will import “Flask” and “request” modules
“io”: to handle stream(to reqd the image bytes stream)
“Image”: to handle images
from ultralytics import YOLO : to use the model
Now to be sure that you have all the needed packages to run your model, you have to install all its requirements by executing the following command:

pip install ultralytics==8.0.9

!git clone https://github.com/ultralytics/ultralytics
%pip install -qe ultralytics
then import all packages :

from PIL import Image
from flask import Flask, request
from ultralytics import YOLO
2. Load your model in your Python code
Now we will load our pre-trained image object detection model in our python code :

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
Note that we will use “yolov8n” and “n” for nano: it means that we will use the small version of yoloV8(it will use small computation power but will be less accurate than the yolov8 larger one).

Let’s load an image and test our model to check if everything is OK :

results = model.predict(
   source='https://media.roboflow.com/notebooks/examples/dog.jpeg')
print("Bounding Boxes :",results[0].boxes.xyxy)
print("Classes :",results[0].boxes.cls)
With this code we load an image and make a prediction, the result is :

YOLOV8 detection results with Python
YOLOV8 detection results with Python
we are getting an array of objects detected in this image, each object has :

xmin, ymin, xmax, ymax: coordinates of the bounding box
Class: the class of the object
Awesome, it is working!

Note : To get the class names dictionary you could run :

model.model.names
{0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}
3. Set up your API with Python Flask
First, you need to create your API :

app = Flask(__name__)
Note: to install Flask you need to run :

pip install flask
Now define the path that your users will use to send you a request(in this case the user will send you an image and you will send him back an array of the objects detected in the image)

@app.route("/objectdetection/", methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        return {"result": "ok"}
Here we’ve set up a POST method to get the user request through “/objectdetection/” path. Then we are getting the image in the variable “file” and giving back as a result {“result”: “ok”} (just to test if everything is OK with our API)

Trick : Now to be able to get access to your API you need to install pyngrok and then import it into your Colab Notebook:

!pip install nest-asyncio pyngrok
Now we have to launch the API :

import io
from PIL import Image
from flask import Flask, request
import nest_asyncio
from pyngrok import ngrok

app = Flask(__name__)

@app.route("/objectdetection/", methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        return {"result": "ok"}


ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
app.run(host="0.0.0.0", port=8000)

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | ** http://127.0.0.1:8000/**.  |

## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)






