project-for-nota
================
Face Emotion Detection
----------------
Model works using 2 deep learning model:
1. Face Detector(FaceBoxes from https://github.com/cs-giung/face-detection-pytorch) size of 3,974KB(4MB)
> model path : './detectors/faceboxes/weights/FaceBoxes.pth'
3. Emotion Classifier(Customized using pretrained Mobilenet v2 provided from torchvision.models) size of 8,980KB(9MB)
> model path : './best_model.pt'

>Total size : 13MB

You can find all the custom functions in custom_functions.py


Whole code is written in main.ipynb with explanation

