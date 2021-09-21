"""
    data_visualizer.py
    Author: Park Jaehun

    Purpose
        Visualize saved odometry information and camera frames 
        to check if they are saved properly.
"""
#!/usr/bin/env python3

import os
import cv2
import sys
import glob
import argparse
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection demo')
    parser.add_argument('--data_path', help='Path to data.', type=str, default='output')
    parser.add_argument('--display', dest='display', help='Display data visualization', action='store_true')
    args = parser.parse_args()
    return args

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
PI = 3.14159265

def rad2deg(rad):
    while rad > PI:
        rad -= PI
    while rad < -PI:
        rad += PI
    return rad/PI*180

if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    display = args.display

    if not os.path.exists(data_path):
        print("Can't find data path...")
        sys.exit()

    pattern = os.path.join(data_path, '*')

    for seq_dir in glob.glob(pattern):
        seq = seq_dir[pattern.find('*'):].split(os.path.sep)[0]
        print(seq)

        img_dir = os.path.join(data_path, seq, 'img')
        if not os.path.exists(img_dir):
            print("Can't find image path...")
            sys.exit()

        out_img_dir = os.path.join(data_path, seq, 'out_img')
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        odom_fn = os.path.join(data_path, seq, 'odom.txt')

        with open(odom_fn, 'r') as odom_file:
            start_flag = True
            before_theta = 0
            while True:
                line = odom_file.readline()
                if not line:
                    break

                data = line.split(',')
                img_fn = data[0]
                theta = float(data[1])

                if start_flag:
                    before_theta = theta
                    start_flag = False
                gap_theta = theta - before_theta

                frame = cv2.imread(os.path.join(img_dir, img_fn))
                width = frame.shape[1]
                height = frame.shape[0]

                frame = cv2.line(frame, (int(width/2), int(height*0.9)), (int(width/2), int(height*0.9)), blue, 5)
                arrow_end = (int(width/2 - 10*(gap_theta/PI)*width/2), int(height*0.9))
                if arrow_end[0] < 0:
                    arrow_end = list(arrow_end)
                    arrow_end[0] = 1
                    arrow_end = tuple(arrow_end)
                if arrow_end[0] > width:
                    arrow_end = list(arrow_end)
                    arrow_end[0] = width
                    arrow_end = tuple(arrow_end)
                arrow_color = blue
                if arrow_end[0] > width/2:
                    arrow_color = green
                else:
                    arrow_color = red

                frame = cv2.arrowedLine(frame, (int(width/2), int(height*0.9)), arrow_end, arrow_color, 5)
                frame = cv2.putText(frame, 'degree : %.2f'%(rad2deg(gap_theta)),(30, int(height*0.8)), cv2.FONT_HERSHEY_PLAIN, 1, blue, 2, cv2.LINE_AA)

                cv2.imwrite(os.path.join(out_img_dir, img_fn), frame)
                if display:
                    cv2.imshow('odom', frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                before_theta = theta