#!/usr/bin/env python
import rospy
import cv2
import grapof
import flapnet
from tensorflow.python.keras import models
from cv_bridge import CvBridge
import numpy as np
from numpy.linalg import inv

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo

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
ts = grapof.TopicsSubscription()
geo = grapof.Geometry()
fn = flapnet.Functions(shape_img=(64, 64, 3))


def callback(image_msg):
    # define global variable
    global cam_disp
    global cv_image

    # convert image to a compatible format
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_image = cv_image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    cam_disp = fn_preproc.image_preproc(cv_image, target_size)


# ROS init
rospy.init_node('camera_flap_detection', anonymous=True)
disp_sub = rospy.Subscriber("/stereo/disparity/image", Image, callback, queue_size=1, buff_size=100000)
disp_mat_sub = rospy.Subscriber("/stereo/disparity", DisparityImage, ts.disp_callback, queue_size=1, buff_size=20)
caminfo = rospy.Subscriber("/stereo/left/camera_info", CameraInfo,ts.caminfo_callback, queue_size=1, buff_size=5)
gp_3d_pub = rospy.Publisher('/stereo/disparity/grasping_point', PointStamped, queue_size=10)

rate = rospy.Rate(10)

# Load model from file
model_path = '/home/aleks/nn_results/nn_ftw.hdf5'
model = models.load_model(model_path, custom_objects={'bce_dice_loss': fn_losses.bce_dice_loss,
                                                           'dice_loss': fn_losses.dice_loss})
print('Neural Network model loaded from file: {}'.format(model_path))

while not rospy.is_shutdown():

    # Predict labels for input image
    pred = model.predict(cam_disp)[0]
    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    # Resize, binarize and clean the depthmap (all functions wants uint8 images)
    pred_res = cv2.resize(pred, depth_orig, interpolation=cv2.INTER_BITS2)
    retval, pred_bin = cv2.threshold(pred_res, 0.5, 1, cv2.THRESH_BINARY)
    pred_bin_clean = img_proc.clean_disparity_map(pred_bin.astype('uint8'), size_th=7500)

    # Convert grey to color image for coloured dots
    pred_bin_color = cv2.cvtColor(pred_bin_clean.astype('uint8'), cv2.COLOR_GRAY2BGR)

    # Invert prediction to detect background centroid
    pred_inv = cv2.bitwise_not(pred_bin_clean.astype('uint8'))

    # Detects centroids of tissues in the image
    centres = img_proc.find_multiple_centroids(pred_bin_clean.astype('uint8'))

    # Evaluate momentum for background centroid detection
    cX, cY = img_proc.find_single_centroid(pred_inv)

    # Detect optimal grasping points and store them in gp[]
    gp = img_proc.find_grasping_points(pred_bin_clean.astype('uint8'), centres, cX, cY)

    # Print detected features on image
    out_img = img_proc.print_background_centroid(pred_bin_color, cX, cY)
    out_img = img_proc.print_grasping_points(out_img, gp)
    out_img = img_proc.print_tissue_centroids(out_img, centres)

    # Show flap detection features
    cv2.imshow("Flap detection", out_img)
    cv2.waitKey(3)

    # Project grasping point on depth map canvas
    if len(gp) != 0:
        gp_pj = geo.project_on_canvas(point=(gp[0][0], gp[0][1]), offset_x=ts.disp_x_offset, offset_y=ts.disp_y_offset)
        disp = ts.disp_mat[gp_pj[1], gp_pj[0]]
        gp_Z = geo.estimate_distance(ts.foc_len, ts.baseline, disp)
        img_gp_coord = (gp_pj[1], gp_pj[0], 1)
        cam_mat = np.asarray(ts.camera_mat).reshape((3,4))
        world_gp_coord = (np.matmul(inv(cam_mat[:, 0:3]), img_gp_coord)) * gp_Z

        gp_3d = PointStamped()
        gp_3d.point.x = world_gp_coord[0]
        gp_3d.point.y = world_gp_coord[1]
        gp_3d.point.z = world_gp_coord[2]

        gp_3d_pub.publish(gp_3d)

    rate.sleep()
