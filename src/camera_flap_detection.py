#!/usr/bin/env python
import rospy
import cv2
import roslib
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing import image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses

rospy.init_node('camera_flap_detection', anonymous=True)
pred_pub = rospy.Publisher('object_detected_probability', Image, queue_size=1)
target_size = (64, 64)
crop = ((56, 521), (160, 665))

bridge = CvBridge()

plt.figure()
def callback(image_msg):
    # First convert the image to OpenCV image
    global cv_image
    global np_image
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_image = cv_image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    cv_image = cv2.resize(cv_image, target_size)  # resize image
    # cv2.imshow("cazzoculo", cv_image)
    np_image = np.expand_dims(cv_image, axis=0)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


rospy.Subscriber("/stereo/disparity/image", Image, callback, queue_size=1, buff_size=16777216)

model = models.load_model('/home/aleks/nn_results/nn_ftw.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss,
                                                                                'dice_loss': dice_loss})
model.summary()


while not rospy.is_shutdown():

    plt.figure()
    plt.imshow(cv_image)
    plt.show()

    prediction = model.predict(np_image)
    pred_reshape = np.squeeze(prediction, axis=0)
    pred_reshape_square = np.squeeze(np.squeeze(prediction, axis=0))

    plt.figure()
    plt.imshow(pred_reshape_square)
    plt.show()

    print('loop')
    rospy.spin()
