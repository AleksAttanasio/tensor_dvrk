#!/usr/bin/env python
import rospy
import cv2
import grapof
import numpy as np

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
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
fn_post = flapnet.Postprocessing()
img_proc = grapof.ImageProcessing()
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
    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    pred_res = cv2.resize(pred, depth_orig, interpolation=cv2.INTER_BITS2)
    retval, pred_bin = cv2.threshold(pred_res, 0.5, 1, cv2.THRESH_BINARY)
    pred_bin_clean = img_proc.clean_disparity_map(pred_bin.astype('uint8'), size_th=5000)

    pred_bin_color = cv2.cvtColor(pred_bin_clean.astype('uint8'), cv2.COLOR_GRAY2BGR)
    pred_inv = cv2.bitwise_not(pred_bin_clean.astype('uint8'))

    centres = img_proc.find_multiple_centroids(pred_bin_clean.astype('uint8'))
    print('Found {} tissues'.format(len(centres)))
    # Evaluate momentum for background centroid detection
    cX, cY = img_proc.find_single_centroid(pred_inv)

    # Detect optimal grasping points and store them in gp[]
    gp = img_proc.find_grasping_points(pred_bin_clean.astype('uint8'), centres, cX, cY)
    print('Found {} grasping points'.format(len(gp)))
    out_img = img_proc.print_background_centroid(pred_bin_color, cX, cY)

    for i in range(len(gp)):
        out_img = cv2.circle(out_img, (gp[i][0], gp[i][1]), 2, (0, 255, 0), 5)

    out_img = img_proc.print_tissue_centroids(out_img, centres)
    baseline = 0.05

    cv2.imshow("Flap detection", out_img)
    cv2.waitKey(3)

    rate.sleep()
