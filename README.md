# Dogspiracy-machine-learning

## Setup with Anaconda Navigator
https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/

Please add the following packages:

*   Tensorflow
*   Tensorboard
*   Keras
*   Numpy
*   Pathlib
*   pyyaml

Make sure to have Python 3.7 installed.

## Setup in PyCharm

### Add Project Interpreter
Open Interpreter Settings and add Anaconda Interpreter.


![Project Interpreter in PyCharm Setting](assets/project-interpreter.png)

## Run project

Run file ["Main.py"](_Main.py_) to train a model, evaluate a model or predict the dog breed of one image with a model.
You choose the action with the CLI-arguments. Run "Main.py train", "Main.py evaluate" or "Main.py predict".

The model that shall be used & the learning hyper parameters can be changed in ["config.py"](_config.py_).

## Data Set

The train data can be downloaded from https://drive.google.com/file/d/1rRVnOYOhCN6yUc6PmunTn3gxKCtl-MrA/view?usp=sharing

Unzip it and move it to the _data_ folder.

The data is a mix of a part of the stanford dog breeds dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/) and additional dog images from the google image search.

First we tried to only use the full stanford dataset with 200 breed classes, but we discovered that our model was not performing well enough with 200 classes and only ~220 images per class.
Therefore we took the 6 classes with the most images available and added extra images by hand using a google image search crawler.

The train data now contains images of these 6 dog breeds:
* Afghan hound (373 image)
* Bernese mountain dog (466 image)
* Irish wolfhound (302 image)
* Maltese dog (356 image)
* Pomeranian (475 image)
* Samoyed (306 image)

The evaluation data to check a model's performance contains 16 images for each class.
It is located at _data/images/validate_ and is saved in this repository.

There is also one prediction image in _data/images/predict_ that shows a very good looking Maltese dog.
To predict an other image, move it to this directory and delete the old image.


## Results
### Model Summary
![summary](assets/summary.PNG)
### Training and Validation
Adam Optimizer:
- training: grey
- validation: orange

RSMProp Optimizer:
- training: pink
- validation: green

Adam vs. RSMprop Optimizer: epoch accuracy

![adam vs. rsmprop](assets/adam-vs-rsmprop.PNG)

Adam Optimizer loss:

![adam](assets/adam-loss.PNG)

RSMprop Optimizer loss:

![rsmprop](assets/rsmprop-loss.PNG)

### Evaluation
Adam Optimizer:

- model_accuracy: 0.8125

- loss: 1.5272420439869165

RSMProp Optimizer:

- model_accuracy: 0.78125

- loss: 1.426356926560402