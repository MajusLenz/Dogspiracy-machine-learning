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

The train data can be downloaded from _____________________________TODO

Unzip it and move it to _data/dog-breeds_

The data is a mix of a part of the stanford dog breeds dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/) and additional dog images from the google image search.

First we tried to only use the full stanford dataset with 200 breed classes, but we discovered that our model was not performing well enough with 200 classes and only ~220 images per class.
Therefore we took the 6 classes with the most images available and added extra images by hand using a google image search crawler.

The train data now contains images of these 6 dog breeds:
* Afghan hound (373 classes)
* Bernese mountain dog (466 classes)
* Irish wolfhound (302 classes)
* Maltese dog (356 classes)
* Pomeranian (475 classes)
* Samoyed (306 classes)

The evaluation data to check a model's performance contains 16 images for each class.
It is located at _data/images/validate_ and is saved in this repository.

There is also one prediction image in _data/images/predict_ that shows a very good looking Maltese dog.
To predict an other image, move it to this directory and delete the old image.

