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
#define WheelTrack 0.108   // units : m, Distance between the two wheels
#define EncoderCountsPerWheel 374
#define WheelRadius 0.035  // units : m, Mobile robot wheel size
#define Decoding 2
#define DistancePerCount (TwoPI * WheelRadius) / (EncoderCountsPerWheel * Decoding)

long previous_left_encoder_counts = 0;
long previous_right_encoder_counts = 0;
ros::Time current_time, last_time;

struct Odom {
    double x;
    double y;
    double th;
};

Odom odom;
double v_left;          // left motor speed
double v_right;         // right motor speed
double v_th;            // angular velocity of robot

double delta_left;      // no of ticks in left encoder since last update
double delta_right;     // no of ticks in right encoder since last update
double delta_distance;  // distance moved by robot since last update
double delta_x;         // corresponding change in x direction
double delta_y;         // corresponding change in y direction
double delta_th;        // corresponding change in heading
double dt;


void WheelCallback(const geometry_msgs::Vector3::ConstPtr& ticks) {
    current_time = ros::Time::now();

    delta_left = ticks->x - previous_left_encoder_counts;
    delta_right = ticks->y - previous_right_encoder_counts;
    dt = (current_time - last_time).toSec();

    v_left = delta_left * DistancePerCount / dt;
    v_right = delta_right * DistancePerCount / dt;

    delta_distance = 0.5f * (double)(delta_left + delta_right) * DistancePerCount;
    delta_th = (double)(delta_right - delta_left) * DistancePerCount / WheelTrack;

    delta_x = delta_distance * (double)cos(odom.th);
    delta_y = delta_distance * (double)sin(odom.th);

    odom.x  += delta_x;
    odom.y  += delta_y;
    odom.th += delta_th;

    previous_left_encoder_counts    = ticks->x;
    previous_right_encoder_counts   = ticks->y;

    last_time = current_time;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "encoder_odometry_node");
    ros::NodeHandle nh;

    ros::Subscriber motor_encoder_sub = nh.subscribe("/motor_encoder", 10, WheelCallback);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/odometry", 50);
    tf::TransformBroadcaster odom_broadcaster;

    ros::Rate loop_rate(10);

    last_time = ros::Time::now();

    while(ros::ok()) {
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(odom.th);

        geometry_msgs::TransformStamped odom_trans;
        odom_trans.header.stamp = current_time;
        odom_trans.header.frame_id = "odom";
        odom_trans.child_frame_id = "base_link";

        odom_trans.transform.translation.x = odom.x;
        odom_trans.transform.translation.y = odom.y;
        odom_trans.transform.translation.z = 0.0;
        odom_trans.transform.rotation = odom_quat;

        odom_broadcaster.sendTransform(odom_trans);

        nav_msgs::Odometry odom_msgs;
        odom_msgs.header.stamp = current_time;
        odom_msgs.header.frame_id = "odom";

        odom_msgs.pose.pose.position.x = odom.x;
        odom_msgs.pose.pose.position.y = odom.y;
        odom_msgs.pose.pose.position.z = 0.0;
        odom_msgs.pose.pose.orientation = odom_quat;
        odom_msgs.pose.covariance =  boost::assign::list_of(1e-3) (0)   (0)  (0)  (0)  (0)
                                                            (0) (1e-3)  (0)  (0)  (0)  (0)
                                                            (0)   (0)  (1e6) (0)  (0)  (0)
                                                            (0)   (0)   (0) (1e6) (0)  (0)
                                                            (0)   (0)   (0)  (0) (1e6) (0)
                                                            (0)   (0)   (0)  (0)  (0)  (1e3) ;

        odom_msgs.child_frame_id = "base_link";
        odom_msgs.twist.twist.linear.x = delta_x / dt;
        odom_msgs.twist.twist.linear.y = delta_y / dt;
        odom_msgs.twist.twist.angular.z = delta_th / dt;
        odom_msgs.twist.covariance = boost::assign::list_of(1e-3) (0)   (0)  (0)  (0)  (0)
                                                            (0) (1e-3)  (0)  (0)  (0)  (0)
                                                            (0)   (0)  (1e6) (0)  (0)  (0)
                                                            (0)   (0)   (0) (1e6) (0)  (0)
                                                            (0)   (0)   (0)  (0) (1e6) (0)
                                                            (0)   (0)   (0)  (0)  (0)  (1e3) ; 

        odom_pub.publish(odom_msgs);

        ros::spinOnce();
        loop_rate.sleep();
    }
}