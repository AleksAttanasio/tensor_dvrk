import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functools
from tensorflow.python.keras import layers
from PIL import Image
import tensorflow as tf
import datetime
import tensorflow.contrib as tfcontrib
import os
import cv2
from tensorflow.python.keras import losses
from sklearn.model_selection import train_test_split

class Functions:
    def __init__(self, shape_img = (128,128,3), size_batch = 25, epochs = 5):
        self.img_shape = shape_img
        self.batch_size = size_batch
        self.epochs = epochs
        self.experiment_path = "/home/aleks/nn_results/"
        return

    def show_dataset_labels(self, x_train, y_train, display_num=5):

        num_train_examples = len(x_train)

        r_choices = np.random.choice(num_train_examples, display_num)

        plt.figure(figsize=(10, 15))

        for i in range(0, display_num * 2, 2):
            img_num = r_choices[i // 2]
            x_pathname = x_train[img_num]
            y_pathname = y_train[img_num]

            plt.subplot(display_num, 2, i + 1)
            plt.imshow(mpimg.imread(x_pathname))
            plt.title("Original Image")

            example_labels = Image.open(y_pathname)
            label_vals = np.unique(example_labels)

            plt.subplot(display_num, 2, i + 2)
            plt.imshow(example_labels)
            plt.title("Masked Image")

        plt.suptitle("Examples of Images and their Masks")
        plt.show()

    def _process_pathnames(self, fname, label_path):
        # We map this function onto each pathname pair
        img_str = tf.read_file(fname)
        img = tf.image.decode_jpeg(img_str, channels=3)

        label_img_str = tf.read_file(label_path)
        # These are gif images so they return as (num_frames, h, w, c)
        label_img = tf.image.decode_gif(label_img_str)[0]
        # The label image should only have values of 1 or 0, indicating pixel wise
        # object (car) or not (background). We take the first channel only.
        label_img = label_img[:, :, 0]
        label_img = tf.expand_dims(label_img, axis=-1)
        return img, label_img

    def shift_img(self, output_img, label_img, width_shift_range, height_shift_range, img_shape):
        """This fn will perform the horizontal or vertical shift"""
        if width_shift_range or height_shift_range:
            if width_shift_range:
                width_shift_range = tf.random_uniform([],
                                                      -width_shift_range * img_shape[1],
                                                      width_shift_range * img_shape[1])
            if height_shift_range:
                height_shift_range = tf.random_uniform([],
                                                       -height_shift_range * img_shape[0],
                                                       height_shift_range * img_shape[0])
            # Translate both
            output_img = tfcontrib.image.translate(output_img,
                                                   [width_shift_range, height_shift_range])
            label_img = tfcontrib.image.translate(label_img,
                                                  [width_shift_range, height_shift_range])
        return output_img, label_img

    def flip_img(self, horizontal_flip, tr_img, label_img):
        if horizontal_flip:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                        lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                        lambda: (tr_img, label_img))
        return tr_img, label_img

    def _augment(self,
                 img,
                 label_img,
                 resize=None,  # Resize the image to some size e.g. [256, 256]
                 scale=1,  # Scale image e.g. 1 / 255.
                 hue_delta=0,  # Adjust the hue of an RGB image by random factor
                 horizontal_flip=False,  # Random left right flip,
                 width_shift_range=0,  # Randomly translate the image horizontally
                 height_shift_range=0):  # Randomly translate the image vertically
        if resize is not None:
            # Resize both images
            label_img = tf.image.resize_images(label_img, resize)
            img = tf.image.resize_images(img, resize)

        if hue_delta:
            img = tf.image.random_hue(img, hue_delta)

        img, label_img = self.flip_img(horizontal_flip, img, label_img)
        img, label_img = self.shift_img(img, label_img, width_shift_range, height_shift_range, self.img_shape)
        label_img = tf.to_float(label_img) * scale
        img = tf.to_float(img) * scale
        return img, label_img

    def get_baseline_dataset(self,
                             filenames,
                             labels,
                             batch_sz,  # =self.batch_size,
                             preproc_fn,  #=functools.partial(self._augment),
                             threads=5,
                             shuffle=True):
        num_x = len(filenames)
        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=threads)
        if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
            assert batch_sz == 1, "Batching images must be of the same size"

        dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

        if shuffle:
            dataset = dataset.shuffle(num_x)

        # It's necessary to repeat our data for all epochs
        dataset = dataset.repeat().batch(batch_sz)
        return dataset

    def plot_loss(self, history):
        dice = history.history['dice_loss']
        val_dice = history.history['val_dice_loss']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, dice, label='Training Dice Loss')
        plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Dice Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def plot_and_save_loss(self, history, exp_path):
        dice = history.history['dice_loss']
        val_dice = history.history['val_dice_loss']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, dice, label='Training Dice Loss')
        plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Dice Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        exp_folder = os.path.join(self.experiment_path, exp_path)
        plt.savefig(os.path.join(exp_folder, "loss.png"))

        plt.show()

    def plot_predictions(self, model, val_set):
        # Let's visualize some of the outputs
        data_aug_iter = val_set.make_one_shot_iterator()
        next_element = data_aug_iter.get_next()

        # Running next element in our graph will produce a batch of images
        plt.figure(figsize=(10, 20))
        for i in range(5):
            batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
            img = batch_of_imgs[0]
            predicted_label = model.predict(batch_of_imgs)[0]

            plt.subplot(5, 3, 3 * i + 1)
            plt.imshow(img)
            plt.title("Input image")

            plt.subplot(5, 3, 3 * i + 2)
            plt.imshow(label[0, :, :, 0])
            plt.title("Actual Mask")
            plt.subplot(5, 3, 3 * i + 3)
            plt.imshow(predicted_label[:, :, 0])
            plt.title("Predicted Mask")
        plt.suptitle("Examples of Input Image, Label, and Prediction")
        plt.show()

    def plot_and_save_predictions(self, model, val_set, exp_path):
        # Let's visualize some of the outputs
        data_aug_iter = val_set.make_one_shot_iterator()
        next_element = data_aug_iter.get_next()

        # Running next element in our graph will produce a batch of images
        plt.figure(figsize=(10, 20))
        for i in range(5):
            batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
            img = batch_of_imgs[0]
            predicted_label = model.predict(batch_of_imgs)[0]

            plt.subplot(5, 3, 3 * i + 1)
            plt.imshow(img)
            plt.title("Input image")

            plt.subplot(5, 3, 3 * i + 2)
            plt.imshow(label[0, :, :, 0])
            plt.title("Actual Mask")
            plt.subplot(5, 3, 3 * i + 3)
            plt.imshow(predicted_label[:, :, 0])
            plt.title("Predicted Mask")
        plt.suptitle("Examples of Input Image, Label, and Prediction")
        exp_folder = os.path.join(self.experiment_path, exp_path)
        plt.savefig(os.path.join(exp_folder, "prediciton.png"))
        plt.show()


    def generate_train_and_val_ds(self, x_train_filenames, y_train_filenames, x_val_filenames, y_val_filenames):

        tr_cfg = {'resize': [self.img_shape[0], self.img_shape[1]],
                  'scale': 1 / 255.,
                  'hue_delta': 0.1,
                  'horizontal_flip': True,
                  'width_shift_range': 0.2,
                  'height_shift_range': 0.2}

        tr_preprocessing_fn = functools.partial(self._augment, **tr_cfg)

        val_cfg = {'resize': [self.img_shape[0], self.img_shape[1]],
                   'scale': 1 / 255.}

        val_preprocessing_fn = functools.partial(self._augment, **val_cfg)

        train_ds = self.get_baseline_dataset(x_train_filenames,
                                             y_train_filenames,
                                             preproc_fn=tr_preprocessing_fn,
                                             batch_sz=self.batch_size)

        val_ds = self.get_baseline_dataset(x_val_filenames,
                                           y_val_filenames,
                                           preproc_fn=val_preprocessing_fn,
                                           batch_sz=self.batch_size)

        return train_ds, val_ds

    def load_filenames(self, df_train, img_dir, label_dir, test_size=0.2):

        x_train_filenames = []
        y_train_filenames = []

        for img_id in df_train['train']:
            x_train_filenames.append(os.path.join(img_dir, img_id))

        for lab_id in df_train['label']:
            y_train_filenames.append(os.path.join(label_dir, lab_id))

        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
            train_test_split(x_train_filenames, y_train_filenames, test_size=test_size, random_state=42)

        return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames

    def format_e(self,n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

    def save_filename_specs(self, test_size, num_filters, dropout_rate, learning_rate):
        experiment_folder = "nn_"

        if num_filters != 32:
            experiment_folder = experiment_folder + "custom_"
        else:
            experiment_folder = experiment_folder + "std_"

        if dropout_rate > 0:
            experiment_folder = experiment_folder + "dout" + str(dropout_rate * 10) + "_"

        experiment_folder = experiment_folder + "lr" + self.format_e(learning_rate) + "_"
        now = datetime.datetime.now().strftime("%d%B%Y_%I-%M%p")
        experiment_folder = experiment_folder + now

        dir_name = os.path.join(self.experiment_path, experiment_folder)
        model_filename = experiment_folder + ".hdf5"
        model_filename = os.path.join(dir_name, model_filename)

        try:
            # Create target Directory
            os.mkdir(dir_name)
            print("Experiment folder ", dir_name, " ---> CREATED ")
        except FileExistsError:
            print("Directory ", dir_name, " already exists")

        spec_path = os.path.join(dir_name, "specs.txt")

        f = open(spec_path, "w+")
        f.write("img_shape: " + str(self.img_shape[0]) + "\n")
        f.write("batch_size: " + str(self.batch_size) + "\n")
        f.write("testset_size: " + str(test_size) + "\n")
        f.write("dropout_rate: " + str(dropout_rate) + "\n")
        f.write("num_filters: " + str(num_filters) + "\n")
        f.write("learning_rate: " + str(learning_rate) + "\n")
        f.close()

        return experiment_folder, model_filename

class Structure:
    def __init__(self, kernel_size=3, stride=2, pool=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool

    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (self.kernel_size, self.kernel_size), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (self.kernel_size, self.kernel_size), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters, dropout_rate):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((self.pool_size, self.pool_size), strides=(self.stride, self.stride))(encoder)
        dropout = layers.Dropout(dropout_rate)(encoder_pool)

        return dropout, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters, dropout_rate):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(self.stride, self.stride), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        dropout = layers.Dropout(dropout_rate)(decoder)
        decoder = layers.BatchNormalization()(dropout)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (self.kernel_size, self.kernel_size), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (self.kernel_size, self.kernel_size), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def std_model(self, input_shape):

        inputs = layers.Input(shape=input_shape)  # 448,480
        encoder0_pool, encoder0 = self.encoder_block(inputs, 32, 0.25)  # 224,240
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64, 0)  # 112 120
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128, 0.35)  # 56 60
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256, 0)  # 28 30
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512, 0.35)  # 14 15
        center = self.conv_block(encoder4_pool, 1024)  # center
        decoder4 = self.decoder_block(center, encoder4, 512, 0.35)  # 16
        decoder3 = self.decoder_block(decoder4, encoder3, 256, 0)  # 32
        decoder2 = self.decoder_block(decoder3, encoder2, 128, 0.35)  # 64
        decoder1 = self.decoder_block(decoder2, encoder1, 64, 0)  # 128
        decoder0 = self.decoder_block(decoder1, encoder0, 32, 0.35)  # 256
        dropout = layers.Dropout(0.4)(decoder0)
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(dropout)

        return inputs, outputs

    def custom_model(self,input_shape, num_filters=32, dropout_rate = 0.25):

        inputs = layers.Input(shape=input_shape)  # 448,480
        encoder0_pool, encoder0 = self.encoder_block(inputs, num_filters, dropout_rate)  # 224,240
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, num_filters*2, 0)  # 112 120
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, num_filters*4, dropout_rate*1.4)  # 56 60
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, num_filters*8, 0)  # 28 30
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, num_filters*16, dropout_rate*1.4)  # 14 15
        center = self.conv_block(encoder4_pool, num_filters*32)  # center
        decoder4 = self.decoder_block(center, encoder4, num_filters*16, dropout_rate*1.4)  # 16
        decoder3 = self.decoder_block(decoder4, encoder3, num_filters*8, 0)  # 32
        decoder2 = self.decoder_block(decoder3, encoder2, num_filters*4, dropout_rate*1.4)  # 64
        decoder1 = self.decoder_block(decoder2, encoder1, num_filters*2, 0)  # 128
        decoder0 = self.decoder_block(decoder1, encoder0, num_filters, dropout_rate*1.4)  # 256
        dropout = layers.Dropout(dropout_rate*2)(decoder0)
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(dropout)

        return inputs, outputs

class LossFunction:
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss


class Preprocessing:
    def image_preproc(self, img, target_size):
        img = cv2.resize(img, target_size)
        np_image = np.asarray(img)  # read as np array
        np_image = np.expand_dims(np_image, axis=0)  # Add another dimension for tensorflow
        test_img = np.float32(np_image) / 255
        return test_img
