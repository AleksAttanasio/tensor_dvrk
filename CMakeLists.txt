cmake_minimum_required(VERSION 2.8.3)
project(tensor_dvrk)

find_package(
        catkin REQUIRED COMPONENTS
        rospy
        catkin REQUIRED COMPONENTS
        roscpp
        cv_bridge
        image_geometry
        image_transport
)

#include(/opt/opencv-3.4.3/build/OpenCVConfig.cmake)
#set(OpenCV_INCLUDE_DIR ${OPENCV_INCLUDE_DIR})

catkin_package()

catkin_python_setup()

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CURSES_INCLUDE_DIRS}
)

add_executable(color_disparity_map src/color_disparity_streamer.cpp)
target_link_libraries(color_disparity_map ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

