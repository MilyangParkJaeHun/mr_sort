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
from queue import Queue

seq = 'test'
hz = 20
time_threshold = 0.01

record_start = False
record_stop = False
start_flag = True

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
        self.frame_buf = Queue()
        self.frame = [0, None]
        self.start_time = time.time()
        self.fps = 0

        self.odom_buf = Queue()
        self.now_odom = (0, 0, 0, 0)
        self.last_pop_odom = (0, 0, 0, 0)

        self.cap = cv2.VideoCapture(0)

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.out_file = open(os.path.join(self.output_path, 'odom.txt'), 'w')

    def write(self):
        global record_start, record_stop

        if record_start and not record_stop:
            frame = self.frame[1]
            odom = self.now_odom[1:]

            font = cv2.FONT_HERSHEY_SIMPLEX
            fn = os.path.join(self.output_path, 'img', '%06d.jpg'%(self.frame_id))
            # frame = cv2.putText(frame, '%.4f'%(time_gap), (100, 100), font, 1, (0,0,0), 1)
            fps = 1./(time.time() - self.start_time)
            # frame = cv2.putText(frame, 'fps : %.2f'%(fps), (100, 150), font, 1, (0,0,0), 1)
            cv2.imwrite(fn, frame)
            self.start_time = time.time()
            if self.frame_id > 0:
                # print("%s,%.4f,%.6f,%.6f"%('%06d.jpg'%(self.frame_id), self.b_odom_th, self.b_odom_x, self.b_odom_y), file=self.out_file)
                print("%s,%.4f,%.6f,%.6f"%('%06d.jpg'%(self.frame_id), odom[2], odom[0], odom[1]), file=self.out_file)
            self.frame_id += 1

    def capture(self):
        now_stamp = rospy.get_rostime().to_sec()
        ret, frame = self.cap.read()
        after_stamp = rospy.get_rostime().to_sec()
        time_stamp = (now_stamp+after_stamp)/2

        if not ret:
            rospy.logerr("Can't read camera")
            sys.exit(1)

        self.frame_buf.put((now_stamp, frame))

    def match_odom_cam(self):
        cam_stamp = self.frame[0]
        
        print('============================================================')
        print('now          : ', rospy.get_rostime().to_sec())
        print('cam stamp    : ', cam_stamp)
        if self.odom_buf.qsize() == 0:
            return 0

        before_odom = self.last_pop_odom
        print('matching start!!!')
        while not self.odom_buf.empty():
            odom = self.odom_buf.get()
            odom_stamp = odom[0]
            
            print('before_stamp : ', before_odom[0])
            print('now_stamp    : ', odom[0])
            if odom_stamp >= cam_stamp and cam_stamp > before_odom[0]:
                stamp_gap = odom[0] - before_odom[0]
                cam_odom_gap = cam_stamp - before_odom[0]
                ratio = cam_odom_gap / stamp_gap if stamp_gap != 0 else 0

                x = before_odom[1] + (odom[1] - before_odom[1]) * ratio
                y = before_odom[2] + (odom[2] - before_odom[2]) * ratio
                th = before_odom[3] + (odom[3] - before_odom[3]) * ratio

                self.now_odom = (odom_stamp, x, y, th)
                self.last_pop_odom = odom
                
                return 1
            elif before_odom[0] > cam_stamp:
                self.now_odom = before_odom
                self.last_pop_odom = before_odom
                return 1

            before_odom = odom
            print('---------------------------------------------------------')
        self.now_odom = before_odom[1:]
        return 0

    def done_write(self):
        self.out_file.close()

    def update_odom(self, time_stamp, x, y, th):
        self.odom_buf.put((time_stamp, x, y, th))

    def update_frame(self):
        self.frame = self.frame_buf.get()

collector = Collector()

def odomCallback(data):
    global collector, record_start, record_stop

    time_stamp = data.header.stamp.to_sec()
    odom_x = data.pose.pose.position.x
    odom_y = data.pose.pose.position.y
    odom_th = data.pose.pose.position.z

    print('now          : ', rospy.get_rostime().to_sec())
    print('odom         : ', time_stamp)
    collector.update_odom(time_stamp, odom_x, odom_y, odom_th)

def joyCallback(data):
    global collector, record_start, record_stop
    button = data.buttons
    if button[0]: # A button
        print('Write start!!!')
        record_start = True
    if button[1]: # B button
        record_stop = True
        collector.done_write()
        print('Write done!!!')
        rospy.on_shutdown(shutdown_hook)

def cameraCallback(event):
    global collector
    collector.capture()

def writeCallback(event):
    global collector, start_flag
    matching_success = collector.match_odom_cam()
    if matching_success or start_flag:
        collector.update_frame()
        collector.write()
        start_flag = False
    else:
        print('Not matching!!!!!!!!!!!!!!!')
        print('Not matching!!!!!!!!!!!!!!!')
        print('Not matching!!!!!!!!!!!!!!!')
        print('Not matching!!!!!!!!!!!!!!!')
        print('Not matching!!!!!!!!!!!!!!!')
    

def shutdown_hook():
    global collector
    sys.exit("Exit")
    rospy.signal_shutdown("Shutdown rosnode")

if __name__ == "__main__":
    rospy.init_node('data_collector', anonymous=True)

    rospy.Subscriber('/odometry', Odometry, odomCallback)
    rospy.Subscriber('/joy', Joy, joyCallback)

    rospy.Timer(rospy.Duration(1/20), cameraCallback)
    rospy.Timer(rospy.Duration(1./hz), writeCallback)

    rospy.spin()
