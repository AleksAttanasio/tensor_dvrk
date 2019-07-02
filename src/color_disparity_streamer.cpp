
#include "color_disparity_streamer.h"

using namespace std;

int main (int argc, char** argv) {

    // ROS init, subs and pubs
    string left_img_topic = "/stereo/left/image_rect_color";
    string right_img_topic = "/stereo/right/image_rect_color";

    ros::init(argc, argv, "color_disparity_streamer");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber left_img_sub = it.subscribe(left_img_topic, 1, &leftImageCallback);
    image_transport::Subscriber right_img_sub = it.subscribe(right_img_topic, 1, &rightImageCallback);
    image_transport::Publisher pub_disp_map = it.advertise("/stereo/disparity/image", 1);
    ros::Subscriber sub_disp_map = nh.subscribe("/stereo/disparity", 1, &disparityCallback);
    ros::Rate loop_rate(100);


    // ROS loop
    while (ros::ok()) {

        // get disparity image as ROS Image message and publish it on a topic
        sensor_msgs::ImagePtr disp_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", disparity_color_).toImageMsg();
        pub_disp_map.publish(disp_img);

        loop_rate.sleep();
        ros::spinOnce();

    }

    return 0;
}