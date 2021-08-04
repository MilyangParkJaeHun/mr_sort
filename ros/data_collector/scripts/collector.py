#!/usr/bin/env python3
import rospy
import rospkg
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from sensor_msgs.msg import Joy

import cv2
import numpy as np
import time
import os
import sys

seq = 'test'

start_flag = False
stop_flag = False

class Collector:
    def __init__(self):
        global seq
        rospack = rospkg.RosPack()
        pack_path = rospack.get_path('data_collector')

        self.output_path = os.path.join(pack_path, 'scripts', 'output', seq)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(os.path.join(self.output_path, 'img')):
            os.makedirs(os.path.join(self.output_path, 'img'))

        self.frame_id = 0
        self.last_time = time.time()

        self.cap = cv2.VideoCapture(0)

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.out_file = open(os.path.join(self.output_path, 'odom.txt'), 'w')

    def write(self, x, y, th):
        global start_flag, stop_flag
        frame = self.get_frame()

        time_gap = time.time() - self.last_time
        if start_flag and not stop_flag and time_gap > 0.065:
            self.frame_id += 1

            fn = os.path.join(self.output_path, 'img', '%06d.jpg'%(self.frame_id))
            cv2.imwrite(fn, frame)

            print("%s,%.4f,%.6f,%.6f"%('%06d.jpg'%(self.frame_id), th, x, y), file=self.out_file)

            self.last_time = time.time()

    def done_write(self):
        self.out_file.close()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Can't load camera")
            return None
        pass

collector = Collector()

def odomCallback(data):
    global collector, start_flag, stop_flag

    odom_x = data.pose.pose.position.x
    odom_y = data.pose.pose.position.y
    odom_th = data.pose.pose.position.z

    collector.write(odom_x, odom_y, odom_th)

def joyCallback(data):
    global collector, start_flag, stop_flag
    button = data.buttons
    if button[0]: # A button
        print('Write start!!!')
        start_flag = True
    if button[1]: # B button
        stop_flag = True
        collector.done_write()
        print('Write done!!!')
        rospy.on_shutdown(shutdown_hook)
    
def shutdown_hook():
    global collector
    sys.exit("Exit")
    rospy.signal_shutdown("Shutdown rosnode")

if __name__ == "__main__":
    rospy.init_node('data_collector', anonymous=True)

    rospy.Subscriber('/odometry', Odometry, odomCallback)
    rospy.Subscriber('/joy', Joy, joyCallback)


    rospy.spin()
