"""
    collector.py
    Author: Park Jaehun

    Purpose
        Synchronize and collect odometry information and camera frames.
"""

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
import threading
from queue import Queue
from collections import deque

seq = 'test'
hz = 20
time_threshold = 0.01

start_flag = False
stop_flag = False

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(3, 640)
    self.cap.set(4, 480)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    self.q = Queue(maxsize=1)
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
    
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
        self.now_frame = (0, None)
        self.start_time = time.time()
        self.fps = 0

        self.odom_buf = deque()
        self.before_odom = np.zeros(4)
        self.last_odom = np.zeros(4)
        self.cam_odom = np.zeros(3)

        self.cap = VideoCapture(0)
        self.out_file = open(os.path.join(self.output_path, 'odom.txt'), 'w')

    def write(self):
        global start_flag, stop_flag

        if start_flag and not stop_flag:

            frame = self.now_frame[1]
            odom = self.last_odom

            font = cv2.FONT_HERSHEY_SIMPLEX
            fn = os.path.join(self.output_path, 'img', '%06d.jpg'%(self.frame_id))

            fps = 1./(time.time() - self.start_time)
            # frame = cv2.putText(frame, 'fps : %.2f'%(fps), (100, 150), font, 1, (0,0,0), 1)
            cv2.imwrite(fn, frame)
            self.start_time = time.time()
            if self.frame_id > 0:
                print("%s,%.4f,%.6f,%.6f"%('%06d.jpg'%(self.frame_id), odom[3], odom[1], odom[2]), file=self.out_file)

            self.frame_id += 1

    def sync_cam_odom(self):
        if len(self.odom_buf) < 2:
            return 0
        cam_stamp = self.now_frame[0]
        last_odom = self.odom_buf.pop()
        before_odom = self.odom_buf.pop()

        while last_odom[0] == before_odom[0] and len(self.odom_buf) > 0:
            before_odom = self.odom_buf.pop()
        
        if before_odom[0] != last_odom[0]:
            self.before_odom = before_odom
            self.last_odom = last_odom

            self.odom_buf.clear()
            self.odom_buf.append(before_odom)
            self.odom_buf.append(last_odom)

            return 1
        else:
            return 0

    def capture(self):
        now_stamp = rospy.get_rostime().to_sec()
        frame = self.cap.read()
        self.now_frame = (now_stamp, frame)

    def done_write(self):
        self.out_file.close()

    def update_odom(self, time_stamp, x, y, th):
        self.odom_buf.append((time_stamp, x, y, th))

collector = Collector()

def odomCallback(data):
    global collector, start_flag, stop_flag

    time_stamp = data.header.stamp.to_sec()
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

    match_success = collector.sync_cam_odom()
    if match_success:
        collector.write()

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

    rospy.Timer(rospy.Duration(1./hz), cameraCallback)
    rospy.Timer(rospy.Duration(1./hz), writeCallback)

    rospy.spin()
