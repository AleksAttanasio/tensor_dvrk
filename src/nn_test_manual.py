import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

import os
import flapnet
import numpy as np
import matplotlib as mpl
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import models
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

model_path = '/home/aleks/nn_results/nn_ftw.hdf5'
testset_size = 0.1

fn = flapnet.Functions(shape_img=(64, 64, 3))
fn_struct = flapnet.Structure()
fn_losses = flapnet.LossFunction()


# Da/home/aleks/nn_results/Gtaset folders init
dataset_name = os.path.join('dataset', 'dataset_ready_aug_02')
img_dir = os.path.join(dataset_name, "train")
label_dir = os.path.join(dataset_name, "labels")
df_train = pd.read_csv(os.path.join(dataset_name,'ready_dataset_aug_02.csv'))

# Load filenames of labels and tra ining objects
x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = fn.load_filenames(df_train= df_train,
                                                                                           img_dir= img_dir,
                                                                                           label_dir= label_dir,
                                                                                           test_size= testset_size)

# Generating dataset for training
train_ds, val_ds = fn.generate_train_and_val_ds(x_train_filenames, y_train_filenames, x_val_filenames, y_val_filenames)

model = models.load_model(model_path, custom_objects={'bce_dice_loss': fn_losses.bce_dice_loss,
                                                           'dice_loss': fn_losses.dice_loss})
model.summary()


fn.plot_predictions(model, val_ds)

