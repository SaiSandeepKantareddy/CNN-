'''This .py file performs the necessary imports to start the training process'''

# Importing the necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import load_training_dataset

# Output path to save the model after training
target_path = r"/home/saikantareddy/Documents/eAI_Generator/eAI_generator_integrated_code/model"

# Learning rate
learning_rate = 0.0001

# Number of epochs
num_epochs = 100

# Batch size
batch_size = -1
training_batch_size = 32

# Image size
img_size = 64

# Number of input channels
num_channels = 3

# Path of the training dataset
train_path = r"/home/saikantareddy/Documents/tutorial-2-image-classifier/train"

# Classes
classes = ['7', '6', '1', '4', '0', '9', '8', '2', '3', '5']
num_classes = len(classes)

# Validation size
validation_size = 0.2

# Loading the training dataset
data = load_training_dataset.read_train_sets(r"/home/saikantareddy/Documents/tutorial-2-image-classifier/train", classes, num_channels, validation_size=validation_size)

print("Number of files in Training-set:{}", len(data.train.labels))
print("Number of files in Validation-set:{}", len(data.valid.labels))
session = tf.Session()
