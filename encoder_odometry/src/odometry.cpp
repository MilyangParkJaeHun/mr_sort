#include <ros/ros.h>

#include <stdlib.h>
#include <unistd.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <cmath>
#include <boost/assign/list_of.hpp>

#define PI 3.14159265
#define TwoPI 6.28318531
#define wheeltrack 0.3876   // units : m, Distance between wheels
#define EncoderCountsPerWheel 396
#define wheelradius 0.0575  // units : m, Mobile robot wheel size


void WheelCallback(const geometry_msgs::Vector3::ConstPtr& ticks) {

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "odometry_publisher");
    ros::NodeHandle nh;

    ros::Subscriber motor_encoder_sub = nh.subscribe("/motor_encoder", 10, WheelCallback);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);
    tf::TransformBroadcaster odom_broadcaster;

    ros::Rate loop_rate(10);

    while(ros::ok()) {
        ROS_INFO("chk!!!");

        ros::spinOnce();
        loop_rate.sleep();
    }
}