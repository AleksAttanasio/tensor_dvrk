#!/usr/bin/env python
import rospy
import cv2
import grapof
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

    # pred_grey = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    #
    # pred_res = cv2.resize(pred, depth_orig, interpolation=cv2.INTER_BITS2)
    # retval, pred_bin = cv2.threshold(pred_res, 0.8, 1, cv2.THRESH_BINARY)
    # pred_8 = pred_bin.astype('uint8') * 255
    # pred_inv = cv2.bitwise_not(pred_8)
    #
    # centres = img_proc.find_multiple_centroids(pred_8)
    #
    # # Evaluate momentum for background centroid detection
    # cX, cY = img_proc.find_single_centroid(pred_inv)
    #
    # # Detect optimal grasping points and store them in gp[]
    # gp = img_proc.find_grasping_points(pred_bin, centres, cX, cY)
    #
    # out_img = img_proc.print_background_centroid(pred_color, cX, cY)
    # out_img = img_proc.print_tissue_centroids(out_img, centres)
    # out_img = img_proc.print_grasping_points(out_img, gp)

    canvas_depth = fn_post.make_canvas_fit(pred)

    cv2.imshow("Flap detection", canvas_depth)
    cv2.waitKey(3)

    rate.sleep()
