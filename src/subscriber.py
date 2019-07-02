#!/usr/bin/env python
import rospy

from std_msgs.msg import String
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image

def callback(data):
    rospy.loginfo( "Image size: %d x %d", data.image.height, data.image.width)


def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/stereo/disparity", DisparityImage, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
