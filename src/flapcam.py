from tensorflow.python.keras import models
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import rospy
import cv2
import roslib

target_size = (64, 64)
crop = ((56, 521), (160, 665))