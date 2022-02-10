# Utility Pole Survey Image Classifier

## TL;DR
This project implements a Deep Convolutional Neural Network (DCNN) to classify utility pole survey images. An overall accuracy of over 96% was achieved on a test dataset.

## Contents
1. [Report](#Report)
    1. [Problem Overview and Definition](#Problem-Overview-and-Definition)
    3. [Data Aggregation, Cleaning, and Dataset Creation](#Data-Aggregation-Cleaning-and-Dataset-Creation)
    4. [Model Exploration and Choice](#Model-Exploration-and-Choice)
    5. [Training and Hyperparameter Tuning](#Training-and-Hyperparameter-Tuning)
    7. [Model Evaluation](#Model-Evaluation)
2. [Usage](#-Usage)
    1. [Getting Started](#Getting-Started)
    2. [Training](#Training)
    3. [Evaluation](#Evaluation)
    4. [Inference](#Interence)

# Report
### Problem Overview and Definition
Data collection and analysis of utility poles is common in the world of Outside Plant (OSP) design. The most efficient OSP engineering companies survey utility poles by photographing them and later employ various processing techniques to extract useful data. Often the first step in this process is to group images by type, which is usually done manually and can be time intensive.

This project aims to develop a model that will classify a survey image into one of six possible classes. An image of each class can be seen in Figure 1.

![Class Examples](/README_imgs/pole_image_class_examples.jpg)

An accuracy of over 90% for each class will be considered a success and a model of lesser complexity is considered better than one of higher complexity. 

### Data Aggregation, Cleaning, and Dataset Creation

1000 images of each class are collected and labeled for a total of 6000 survey images. At this point, all images are located in one directory and their filenames contain their class name at some point. Now it is easy to write a script to group the images into individual class folders.

Finally, the class-split data is further "randomly" split into training, validataion, and test datasets. This operation can be seen in the code block below.

```py
import splitfolders
data_dir = "path/to/class/split/data"
split_dir = "path/to/desired/trainvaltest/directory"
splitfolders.ratio(data_dir, output=split_dir, seed=1337, ratio=(0.5, 0.1, 0.4), group_prefix=None)
```

Given the defined split ratio above, each class will have 500 images in the training set, 100 in the validation set, and 400 in the test set.

### Model Exploration and Choice
Considering the problem domain and previous experience, it is clear that a CNN is the most promising approach. Therefore, a popular CNN architecture that is available as a pretrained model is choosen. A pretrained model like this can tuned to a custom dataset as long as the custom dataset lies in a subdomain of the dataset in which the pretrained model was originally trained. This is known as transfer learning. The [DenseNet-121](https://arxiv.org/abs/1608.06993) model is chosen for the first attempt as it is best to start with a simple model and increase complexity from there.

In order to make sure that a proposed model is actually capable of learning from a dataset, it is good practice to attempt and overfit this model to one batch of training data. The learning curve of the described experiment can be seen below.

![Overfit Learning Curve](/README_imgs/overfit_single_batch.png)

As can be seen, the loss reaches near-zero values. For this portion of the project, this is a good sign, the proposed model is capable of learning from the custom dataset. Full steam ahead with the DenseNet-121!

### Training and Hyperparameter Tuning

Now it is time to squeeze every bit of performance out of the chosen architecture. There are various parameters that will have to be chosen in order to achieve the best configuration, which are called hyperparameters. To find a near-optimal permutation of hyperparameters, multiple models are trained with different hyperparameters and the best performing model is chosen. The random search algorithm is used and is implemented using [Ray Tune](https://docs.ray.io/en/master/tune/index.html) library. Only batch size and learning rate are treated as hyperparameters and the search space is configured as seen in the code below and found [here](https://github.com/shankal17/DenseNet-Utility-Pole-Survey-Image-Classifier/blob/main/tune.py#:~:text=config%20%3D%20%7B,%7D).

```py
# Define hyperparameter search space
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8, 16, 32])
}
```

Summarized results from the tuning process can be seen below

![Hyperparam Summary](/README_imgs/tuning_results.PNG)
![Best Config](/README_imgs/optimal_tuning_result.PNG)

Most of the training runs are terminated early, as they show little promise but the best model achieved over 97% accuracy when run on a small validation dataset. This is the model that is ultimately chosen.

### Model Evaluation

The model is trained, now it is time to conduct inference on a larger test dataset i.e. images that were not "seen" in training. The overall accuracy and individual class accuracy is reported in a log file.

![Best Config](/README_imgs/final_model_evaluation_log_screenshot.PNG)

That exceeds the minimum requirements to be considered successfull!

# Usage
### Getting Started
Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```
### Training
All that needs to be done to train/tune the model is to run the [tune.py](https://github.com/shankal17/DenseNet-Utility-Pole-Survey-Image-Classifier/blob/main/tune.py) model, changing the appropriate directory paths and configurations. Changing the search space is as simple as changing the numbers in the code block below, also found [here](https://github.com/shankal17/DenseNet-Utility-Pole-Survey-Image-Classifier/blob/main/tune.py#:~:text=config%20%3D%20%7B,%7D).

```py
# Define hyperparameter search space
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8, 16, 32])
}
```

### Evaluation
Once a model is trained, just run the [evaluate.py](https://github.com/shankal17/DenseNet-Utility-Pole-Survey-Image-Classifier/blob/main/evaluate.py) module, being sure to change the appropriate lines in the code block below, found [here](https://github.com/shankal17/DenseNet-Utility-Pole-Survey-Image-Classifier/blob/main/evaluate.py#:~:text=if%20__name__%20%3D%3D%20%27__main__,classes%2C%20data_dir%3Ddata_dir).

```py
if __name__ == '__main__':
    data_dir = 'data/processed/train_val_split'
    classes = ('Birthmark', 'Grounded', 'Height Shot', 'Midspan', 'Pole Tag', 'Upshot')
    model_path = 'path/to/model/checkpoint'
    
    evaluate(model_path, classes, data_dir=data_dir)
```

### Inference
You can integrate this into a bunch of different systems or change it to make decisions based on how "confident" (sigmoid output) the model is.


