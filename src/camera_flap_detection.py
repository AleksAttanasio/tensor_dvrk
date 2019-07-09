#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import flapnet
from tensorflow.python.keras import models

# Manual settings
bridge = CvBridge()
orig_size = (720, 576)
depth_orig = (506,466)
target_size = (64, 64)
crop = ((55, 521), (159, 665))

# Create Flapnet objects
fn_struct = flapnet.Structure()
fn_losses = flapnet.LossFunction()
fn_preproc = flapnet.Preprocessing()
fn = flapnet.Functions(shape_img=(64, 64, 3))


def callback(image_msg):
    # declare global variable
    global cam_disp
    global cv_image

    # convert image to a compatible format
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_image = cv_image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    cam_disp = fn_preproc.image_preproc(cv_image, target_size)
    # cv2.imshow("Input Depthmap", cv_image)
    # cv2.waitKey(3)


# ROS init
rospy.init_node('camera_flap_detection', anonymous=True)
pred_pub = rospy.Publisher('nn/prediction/mask', Image, queue_size=1)
disp_sub = rospy.Subscriber("/stereo/disparity/image", Image, callback, queue_size=1, buff_size=100000)
rate = rospy.Rate(10)

# Load model from file
model_path = '/home/aleks/nn_results/nn_ftw.hdf5'
model = models.load_model(model_path, custom_objects={'bce_dice_loss': fn_losses.bce_dice_loss,
                                                           'dice_loss': fn_losses.dice_loss})
print('Neural Network model loaded from file: {}'.format(model_path))

while not rospy.is_shutdown():

    pred = model.predict(cam_disp)[0]
    pred_res = cv2.resize(pred, depth_orig, interpolation=cv2.INTER_CUBIC)
    canvas_depth = np.zeros((576,720), np.float32)
    canvas_depth[55:521, 159:665] = pred_res



    cv2.imshow("Flap detection", canvas_depth)
    cv2.waitKey(3)

    rate.sleep()
