import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

import cv2
from PIL import Image
import os
import flapnet
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import models
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

fn = flapnet.Functions(shape_img=(64, 64, 3))
fn_struct = flapnet.Structure()
fn_losses = flapnet.LossFunction()
fn_preproc = flapnet.Preprocessing()

model_path = '/home/aleks/nn_results/nn_ftw.hdf5'
file_path = '/home/aleks/catkin_ws/src/tensor_dvrk/src/dataset/data_dummy/disp_lobe2_05008.jpeg'

target_size = (64, 64)
img = cv2.imread(file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test_img = fn_preproc.image_preproc(img, target_size)

# Generating dataset for training
model = models.load_model(model_path, custom_objects={'bce_dice_loss': fn_losses.bce_dice_loss,
                                                           'dice_loss': fn_losses.dice_loss})
# model.summary()
print('Neural Network model loaded')

pred = model.predict(test_img)[0]

print('Input image shape {}'.format(test_img.shape))
print('Prediction shape: {}'.format(pred.shape))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input image")
plt.subplot(1, 2, 2)
plt.imshow(pred[:, :, 0])
plt.title("Predicted Label")
plt.show()

