"""
Visualize MOT GT data using OpenCV
MOT GT data format  : [frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]
"""
import os

from numpy.lib.function_base import blackman
from scipy.optimize.zeros import _within_tolerance
import cv2
import csv
import glob
import random
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Visualize GT data")
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='/home/openvino/dev/mr_sort/tracker/data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')

    args = parser.parse_args()
    return args

def parse_gt(line):
    data = line.split(',')

    bbox = dict()
    bbox["img"]         = '%06d.jpg'%(data[0])
    bbox["id"]          = data[1]
    bbox["xmin"]        = data[2]
    bbox["ymin"]        = data[3]
    bbox["xmax"]        = data[2] + data[4]
    bbox["ymax"]        = data[3] + data[5]
    bbox["conf"]        = data[6]
    bbox["class"]       = data[7]
    bbox["visibilty"]   = data[8]

    return bbox

def parse_img(list):
    bbox_list = dict()


blue    = (255, 0, 0)
green   = (0, 255, 0)
red     = (0, 0, 255)
yellow  = (0, 255, 255)
black   = (255, 255, 255)
white   = (0, 0, 0)
font    = cv2.FONT_HERSHEY_SIMPLEX

color_map = dict()

for i in range(1000):
    r = random.randrange(1, 255)
    g = random.randrange(1, 255)
    b = random.randrange(1, 255)
    color_map[i] = (b, g, r)

if __name__ == "__main__":
    args = parse_args()
    seq_path = args.seq_path
    phase = args.phase

    pattern = os.path.join(seq_path, phase, '*')
    print(pattern)
    for seq_dir in glob.glob(pattern):
        seq = seq_dir[pattern.find('*'):].split(os.path.sep)[0]

        gt_fn = os.path.join(seq_dir, 'gt', 'gt.txt')
        if not os.path.exists(gt_fn):
            print("Can't find gt file...")

        bbox_list = {}
        with open(gt_fn, 'r') as in_file:
            while True:
                line = in_file.readline()
                if not line:
                    break

                data = line.split(",")
                if not data[0] in bbox_list.keys():
                    bbox_list[data[0]] = [data[1:]]
                else:
                    bbox_list[data[0]].append(data[1:])

            person_count = 0
            for img_id in bbox_list.keys():
                img_fn = '%.6d.jpg'%(int(img_id))
                frame = cv2.imread(os.path.join(seq_dir, 'img1', img_fn))
                # print(os.path.join(seq_dir, 'img1', img_fn))

                for bbox in bbox_list[img_id]:
                    identity    = int(bbox[0])
                    xmin        = int(bbox[1])
                    ymin        = int(bbox[2])
                    xmax        = int(bbox[1]) + int(bbox[3])
                    ymax        = int(bbox[2]) + int(bbox[4])
                    conf        = int(bbox[5])
                    class_id    = int(bbox[6])
                    visibility  = float(bbox[7])

                    if visibility < 0.5:
                        continue
                    person_count += 1
            print(gt_fn)
            print(person_count)
            print('-----------------------------------------------')
