#!/usr/bin/env python3

import os
import cv2
import sys
import glob
import argparse
import numpy as np
import time
from DetModel import OpenvinoDet
from yolov4 import Yolov4 as Yolo
from ssd import SsdMobilenet as Ssd

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection demo')
    parser.add_argument('--model_type', help='Type of object detection model : [ssd / yolo]', type=str, default='')
    parser.add_argument('--model_path', help='Path to object detection model weight file', type=str, default='IR/Yolo/coco')
    parser.add_argument('--img_path', help='Path to input images.', type=str, default='../mot_benchmark')
    parser.add_argument('--device', help='Device for inference', type=str, default='GPU')
    parser.add_argument("--prob_threshold", help='Minimum probability for detection.', type=float, default=0.5)
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument('--phase', help="Subdirectory in seq_path.", type=str, default='train')
    args = parser.parse_args()
    return args   

if __name__ == "__main__":
    args = parse_args()

    display = args.display
    phase = args.phase

    label_map = [''] * 90
    label_map[0] = 'person'
    if args.model_type == "yolo":
        detector = Yolo(args.model_path, args.device, label_map, args.prob_threshold)
    elif args.model_type == "ssd":
        detector = Ssd(args.model_path, args.device, label_map, args.prob_threshold)
    else:
        print("Supports ssd or yolo as detection models. \n Choose between ssd and yolo!!!")
        sys.exit(1)

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.img_path, phase, '*', 'img1')


    total_time = 0 
    total_frame = 0
    for img_dir in glob.glob(pattern):
        detector.clear()

        frame_id = 0
        seq = img_dir[pattern.find('*'):].split(os.path.sep)[0]
        out_det_path = os.path.join(args.img_path, phase, seq, 'det')
        if not os.path.exists(out_det_path):
            os.makedirs(out_det_path)

        out_img_dir = os.path.join('output', seq, 'img')
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        with open(os.path.join(out_det_path, 'det.txt'), 'w') as out_file:

            img_id_list = [i+1 for i in range(len(os.listdir(img_dir)))]
            img_id_list.append(img_id_list[-1])
            for img_id in img_id_list:
                frame_id += 1
                img_file = '%06d.jpg'%(img_id)

                img_file_path = os.path.join(img_dir, img_file)
                print(img_file_path)
                frame = cv2.imread(img_file_path, cv2.IMREAD_COLOR)

                det_start_time = time.time()
                res = detector.inference(frame)
                det_end_time = time.time()

                total_time += det_end_time - det_start_time
                total_frame += 1

                before_frame_id = frame_id - 1
                for object_dict in res:
                    if not object_dict['class_id'] == 0:
                        continue

                    d = detector.to_dets(object_dict)

                    print('%d,-1,%d,%d,%d,%d,1,-1,-1,-1'\
                            %(before_frame_id, d['bb_left'], d['bb_top'], d['bb_width'], d['bb_height']),file=out_file)

                if display:
                    out_img_path = os.path.join(out_img_dir, '%06d.jpg'%(before_frame_id))
                    out_frame = detector.get_results_img(res)

                    if hasattr(out_frame, 'size') and frame_id > 1:
                        cv2.imwrite(out_img_path, out_frame)

                    detector.update_frame(frame)
    print('fps : ', total_frame / total_time)
                    




