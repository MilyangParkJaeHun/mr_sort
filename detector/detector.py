#!/usr/bin/env python3

import os
import cv2
import sys
import glob
import argparse
import numpy as np
from yolo import Yolo

class Detector(object):
    def __init__(self, model_path, device, prob_threshold):
        """
        Set key parameters for Detector
        """
        self.label_map = [''] * 90
        self.label_map[0] = 'person'
        self.prob_threshold = prob_threshold

        self.model = Yolo(model_path, device, self.label_map, self.prob_threshold)
        self.before_frame = None

    def inference(self, frame):
        """
        Get the infrence results from Object Detection Model
        
        Because the Object Detection Model works asynchronously, 
        the current result is the detection result of the previous frame
        
        output : [[class_id, xmin, ymin, xmax, ymax, confidence], ... ]
        """
        return self.model.inference(frame)

    def clear(self):
        """
        Reload Object Detecton Model to refresh detection results buffer
        """
        return self.model.reload()

    def draw_results(self, res):
        """
        Draw the infrerence results from Object Detection Model on frame
        """
        if hasattr(self.before_frame, 'size'):
            return self.model.draw_objects(self.before_frame, res)
    
    def update_frame(self, frame):
        """
        Update before frame to current frame
        """
        self.before_frame = frame.copy()
    
    def get_results(self, res):
        """
        Get the detection result drawn over the frame

        Since object detection is performed asynchronously, 
        the current detection result is drawn in the previous frame.

        output : frame
        """
        self.draw_results(res)

        return self.before_frame

    def to_dets(self, dict):
        """
        Convert the detection results format 

        from dict format to dets format used for MOTChallengeEvalKit

        dict : [class_id, xmin, ymin, xmax, ymax, confidence]
        dets : [frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z]
        """
        dets = {}
        dets['bb_left']     = dict['xmin']
        dets['bb_top']      = dict['ymin']
        dets['bb_width']    = dict['xmax'] - dict['xmin']
        dets['bb_height']   = dict['ymax'] - dict['ymin']
        dets['conf']        = dict['confidence']


        return dets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection demo')
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

    detector = Detector(args.model_path, args.device, args.prob_threshold)

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.img_path, phase, '*', 'img1')

    for img_dir in glob.glob(pattern):
        detector.clear()

        frame_id = 0
        seq = img_dir[pattern.find('*'):].split(os.path.sep)[0]
        out_det_path = os.path.join('output', seq)
        if not os.path.exists(out_det_path):
            os.makedirs(out_det_path)

        out_img_dir = os.path.join('output', seq, 'img')
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        with open(os.path.join(out_det_path, 'det.txt'), 'w') as out_file:
            print('output file : ', out_file)

            img_id_list = [i+1 for i in range(len(os.listdir(img_dir)))]
            img_id_list.append(img_id_list[-1])
            for img_id in img_id_list:
                frame_id += 1
                img_file = '%06d.jpg'%(img_id)

                img_file_path = os.path.join(img_dir, img_file)
                print(img_file_path)
                frame = cv2.imread(img_file_path, cv2.IMREAD_COLOR)

                res = detector.inference(frame)
                before_frame_id = frame_id - 1
                for object_dict in res:
                    if not object_dict['class_id'] == 0:
                        continue

                    d = detector.to_dets(object_dict)

                    print('%d,-1,%d,%d,%d,%d,1,-1,-1,-1'\
                            %(before_frame_id, d['bb_left'], d['bb_top'], d['bb_width'], d['bb_height']),file=out_file)

                if display:
                    out_img_path = os.path.join(out_img_dir, '%06d.jpg'%(before_frame_id))
                    out_frame = detector.get_results(res)

                    if hasattr(out_frame, 'size') and frame_id > 1:
                        cv2.imwrite(out_img_path, out_frame)

                    detector.update_frame(frame)
                    




