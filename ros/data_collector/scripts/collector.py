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
hz = 20
time_threshold = 0.01

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
        self.now_frame = []
        self.next_frame = None
        self.start_time = time.time()
        self.fps = 0

        self.odom_stamp = 0
        self.odom_x = 0
        self.odom_y = 0
        self.odom_th = 0
        self.b_odom_stamp = 0
        self.b_odom_x = 0
        self.b_odom_y = 0
        self.b_odom_th = 0

        self.cap = cv2.VideoCapture(0)

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.out_file = open(os.path.join(self.output_path, 'odom.txt'), 'w')

    def write(self):
        global start_flag, stop_flag

        if start_flag and not stop_flag:
            if len(self.now_frame) == 0:
                self.now_frame.append(self.next_frame)
            
            time_gap = float(self.now_frame[0][0].to_nsec() - self.odom_stamp.to_nsec())/1000000000 # sec
            if abs(time_gap) > time_threshold:
                if time_gap > 0:
                    return
            
            odom_x = self.odom_x
            odom_y = self.odom_y
            odom_th = self.odom_th

            frame = self.now_frame[0][1]
            self.now_frame.pop()

            font = cv2.FONT_HERSHEY_SIMPLEX

            self.frame_id += 1
            fn = os.path.join(self.output_path, 'img', '%06d.jpg'%(self.frame_id))
            frame = cv2.putText(frame, '%.4f'%(time_gap), (100, 100), font, 1, (255,255,255), 1)
            fps = 1./(time.time() - self.start_time)
            frame = cv2.putText(frame, 'fps : %.2f'%(fps), (100, 150), font, 1, (255,255,255), 1)
            cv2.imwrite(fn, frame)
            self.start_time = time.time()
            print("%s,%.4f,%.6f,%.6f"%('%06d.jpg'%(self.frame_id), odom_th, odom_x, odom_y), file=self.out_file)

    def capture(self):
        now_stamp = rospy.get_rostime()
        _, frame = self.cap.read()
        if len(self.now_frame) == 0:
            self.now_frame.append([now_stamp, frame])
        else:
            self.next_frame = [now_stamp, frame]

    def done_write(self):
        self.out_file.close()


    def update_odom(self, time_stamp, x, y, th):
        self.b_odom_stamp = self.odom_stamp
        self.b_odom_x = self.odom_x
        self.b_odom_y = self.odom_y
        self.b_odom_th = self.odom_th

        self.odom_stamp = time_stamp
        self.odom_x = x
        self.odom_y = y
        self.odom_th = th

    # def get_odom(self, now_stamp):
    #     pass
    #     # idx = self.odom_buf_idx
    #     # buf_size = self.odom_buf_size


    #     # min_gap = 987654321
    #     # for _ in range(buf_size):
    #     #     idx = (idx - 1 + buf_size) % buf_size
    #     #     if not hasattr(self.odom_buf[idx], 'keys'):
    #     #         continue
    #     #     stamp = self.odom_buf[idx]['stamp']
            
    #     #     time_gap = abs(now_stamp.to_nsec() - stamp.to_nsec())
    #     #     if time_gap > min_gap:
    #     #         break
    #     #     else:
    #     #         min_gap = time_gap
    #     # return self.odom_buf[(idx+1+buf_size) % buf_size]
    #     # return self.odom_buf[(idx-1+buf_size) % buf_size]

collector = Collector()

def odomCallback(data):
    global collector, start_flag, stop_flag

    time_stamp = data.header.stamp
    odom_x = data.pose.pose.position.x
    odom_y = data.pose.pose.position.y
    odom_th = data.pose.pose.position.z

    collector.update_odom(time_stamp, odom_x, odom_y, odom_th)

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
    
def writeCallback(event):
    global collector

    collector.write()

    # buf = collector.odom_buf
    # buf = sorted(buf, key=lambda x:  (int(x['stamp'].to_nsec() if hasattr(x, 'size') else 0)))
    # print('------------------------------------\n')
    # for odom in buf:
    #     if not hasattr(odom, 'keys'):
    #         continue
    #     print('%.2f '%(odom['stamp'].to_nsec()), end=' ')
    # print(rospy.get_rostime().to_nsec())

def cameraCallback(event):
    global collector
    
    collector.capture()

def shutdown_hook():
    global collector
    sys.exit("Exit")
    rospy.signal_shutdown("Shutdown rosnode")


if __name__ == "__main__":
    rospy.init_node('data_collector', anonymous=True)

    rospy.Subscriber('/odometry', Odometry, odomCallback)
    rospy.Subscriber('/joy', Joy, joyCallback)

    rospy.Timer(rospy.Duration(1/30), cameraCallback)
    rospy.Timer(rospy.Duration(1./hz), writeCallback)

    rospy.spin()
