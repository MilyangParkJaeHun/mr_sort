#!/usr/bin/env python3

from abc import *

import cv2
import numpy as np

class OpenvinoDet(metaclass=ABCMeta):
    """
    Declare key parameters for Detector
    """
    model_path = ""
    device = "cpu"
    label_map = []
    
    model = None
    before_frame = None # for frame synchronization

    @abstractmethod
    def inference(self, frame):
        """
        Get the infrence results from Object Detection Model
        
        Because the Object Detection Model works asynchronously, 
        the current result is the detection result of the previous frame
        
        output : [[class_id, xmin, ymin, xmax, ymax, confidence], ... ]
        """
        print("Detect objects in current frame")

    @abstractmethod
    def clear(self):
        """
        Reload Object Detecton Model to refresh detection results buffer
        """
        print("Clear detection results buffer")

    def update_frame(self, frame):
        """
        Update before frame to current frame
        """
        self.before_frame = frame.copy()

    def get_results_img(self, dets):
        """
        Draw the infrerence results from Object Detection Model on frame

        Since object detection is performed asynchronously, 
        the current detection result is drawn in the previous frame.

        output : frame
        """
        if hasattr(self.before_frame, 'size'):
            for det in dets:
                # color = (min(det['class_id'] * 17, 255),
                #         min(det['class_id'] * 7, 255),
                #         min(det['class_id'] * 5, 255))
                color = (100, 100, 255)
                det_label = self.label_map[det['class_id']] if self.label_map and len(self.label_map) >= det['class_id'] \
                            else str(det['class_id'])

                cv2.rectangle(self.before_frame, (det['xmin'], det['ymin']), (det['xmax'], det['ymax']), color, 2)
                cv2.putText(self.before_frame,
                            "#" + det_label + ' ' + str(round(det['confidence'] * 100, 1)) + ' %',
                            (det['xmin'], det['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)  

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

    def intersection_over_union(self, box_1, box_2):#add DIOU-NMS support
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])

        cw = max(box_1['xmax'], box_2['xmax'])-min(box_1['xmin'], box_2['xmin'])
        ch = max(box_1['ymax'], box_2['ymax'])-min(box_1['ymin'], box_2['ymin'])
        c_area = cw**2+ch**2+1e-16
        rh02 = ((box_2['xmax']+box_2['xmin'])-(box_1['xmax']+box_1['xmin']))**2/4+((box_2['ymax']+box_2['ymin'])-(box_1['ymax']+box_1['ymin']))**2/4

        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union-pow(rh02/c_area,0.6)

    def preprocess_frame(self, frame, n, c, h, w):
        in_frame = cv2.resize(frame, (w, h),
                            interpolation=cv2.INTER_CUBIC)
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        return in_frame
    
    def filter_dets(self, dets, prob_threshold, iou_threshold):
        dets = sorted(dets, key=lambda det : det['confidence'], reverse=True)
        for i in range(len(dets)):
            if dets[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(dets)):
                if self.intersection_over_union(dets[i], dets[j]) > iou_threshold:
                    dets[j]['confidence'] = 0
        return tuple(det for det in dets if (det['confidence'] >= prob_threshold and det['class_id'] == 0))

    def validation_dets(self, dets, img_height, img_width):
        for det in dets:
            det['xmax'] = min(det['xmax'], img_width)
            det['ymax'] = min(det['ymax'], img_height)
            det['xmin'] = max(det['xmin'], 0)
            det['ymin'] = max(det['ymin'], 0)

        return dets