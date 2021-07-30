#!/usr/bin/env python3

import logging
import threading
import os
import sys
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import perf_counter
from enum import Enum
import time

import cv2
import numpy as np
from openvino.inference_engine import IECore
import ngraph as ng

class YoloParams:
  # ------------------------------------------- Extracting layer parameters ------------------------------------------
  # Magic numbers are copied from yolo samples
  def __init__(self, param, side):
    self.num = 3 if 'num' not in param else int(param['num'])
    self.coords = 4 if 'coords' not in param else int(param['coords'])
    self.classes = 80 if 'classes' not in param else int(param['classes'])
    self.side = side
    self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else param['anchors']

    self.isYoloV3 = False

    if param.get('mask'):
        mask = param['mask']
        self.num = len(mask)

        maskedAnchors = []
        for idx in mask:
            maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        self.anchors = maskedAnchors

        self.isYoloV3 = True

class Yolo:
  def __init__(self, model_path, device, label_map, prob_threshold=0.5):
    self.model_xml = model_path+'/frozen_darknet_yolov4_model.xml'
    self.model_bin = model_path+'/frozen_darknet_yolov4_model.bin'
    print(self.model_xml)
    print(self.model_bin)

    self.device_name = device
    print('device  : ', self.device_name)

    self.label_map = label_map

    self.num_requests = 2
    self.cur_request_id = 0
    self.next_request_id = 1
    self.iou_threshold = 0.4
    self.prob_threshold = prob_threshold

    self.origin_im_size = [480, 640]

    print("Reading IR...")
    self.ie = IECore()
    self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
    self.input_blob = next(iter(self.net.inputs))
    self.out_blob = next(iter(self.net.outputs))
    print("Input  : ", self.input_blob)
    print("Output : ", self.out_blob)
    print("Loading IR to the plugin...")
    self.ie.load_network(network=self.net, device_name=self.device_name, num_requests=self.num_requests)
    self.exec_net = self.ie.load_network(network=self.net, device_name=self.device_name, num_requests=self.num_requests)
    print("Successfully loaded!!!")
    self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape

  def reload(self):
    self.exec_net = self.ie.load_network(network=self.net, device_name=self.device_name, num_requests=self.num_requests)
    print("Successfully reload!!!")

  def preprocess_frame(self, frame):
    in_frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
    return in_frame

  def scale_bbox(self, x, y, height, width, class_id, confidence, im_h, im_w):
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

  def parse_yolo_region(self, predictions, resized_image_shape, original_im_shape, params, threshold):
      # ------------------------------------------ Validating output parameters ------------------------------------------
      _, _, out_blob_h, out_blob_w = predictions.shape
      assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                      "be equal to width. Current height = {}, current width = {}" \
                                      "".format(out_blob_h, out_blob_w)

      # ------------------------------------------ Extracting layer parameters -------------------------------------------
      orig_im_h, orig_im_w = original_im_shape
      resized_image_h, resized_image_w = resized_image_shape
      objects = list()
      size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
      bbox_size = params.coords + 1 + params.classes
      # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
      for row, col, n in np.ndindex(params.side, params.side, params.num):
          # Getting raw values for each detection bounding box
          bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
          x, y, width, height, object_probability = bbox[:5]
          class_probabilities = bbox[5:]
          if object_probability < threshold:
              continue
          # Process raw value
          x = (col + x) / params.side
          y = (row + y) / params.side
          # Value for exp is very big number in some cases so following construction is using here
          try:
              width = exp(width)
              height = exp(height)
          except OverflowError:
              continue
          # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
          width = width * params.anchors[2 * n] / size_normalizer[0]
          height = height * params.anchors[2 * n + 1] / size_normalizer[1]

          class_id = np.argmax(class_probabilities)
          confidence = class_probabilities[class_id]*object_probability
          if confidence < threshold:
              continue
          objects.append(self.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence, \
                                    im_h=orig_im_h, im_w=orig_im_w))
      return objects

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

  def filter_objects(self, objects, prob_threshold):
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if self.intersection_over_union(objects[i], objects[j]) > self.iou_threshold:
                objects[j]['confidence'] = 0
    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)

  def validation_objects(self, objects):
    for obj in objects:
      obj['xmax'] = min(obj['xmax'], self.origin_im_size[1])
      obj['ymax'] = min(obj['ymax'], self.origin_im_size[0])
      obj['xmin'] = max(obj['xmin'], 0)
      obj['ymin'] = max(obj['ymin'], 0)

    return objects

  def get_objects(self, output, net, new_frame_height_width, source_height_width, prob_threshold):
    objects = list()
    function = ng.function_from_cnn(net)
    for layer_name, out_blob in output.items():
        out_blob = out_blob.buffer.reshape(net.outputs[layer_name].shape)
        params = [x._get_attributes() for x in function.get_ordered_ops() if x.get_friendly_name() == layer_name][0]
        layer_params = YoloParams(params, out_blob.shape[2])
        objects += self.parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                    prob_threshold)

    return objects

  def switching_request_id(self):
    self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

  def inference(self, frame):
    in_frame = self.preprocess_frame(frame)
    self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})
    objects = list()

    self.origin_im_size[0] = frame.shape[0]
    self.origin_im_size[1] = frame.shape[1]

    if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
        output = self.exec_net.requests[self.cur_request_id].output_blobs
        objects = self.get_objects(output, self.net, (self.h, self.w), (self.origin_im_size[0], self.origin_im_size[1]), 0.5)
        objects = self.filter_objects(objects, self.prob_threshold)
        objects = self.validation_objects(objects)
    self.switching_request_id()
    return objects

  def draw_objects(self, frame, objects):
    for obj in objects:
      color = (min(obj['class_id'] * 17, 255),
              min(obj['class_id'] * 7, 255),
              min(obj['class_id'] * 5, 255))
      det_label = self.label_map[obj['class_id']] if self.label_map and len(self.label_map) >= obj['class_id'] else \
                  str(obj['class_id'])

      cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
      cv2.putText(frame,
                  "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                  (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)      

if __name__ == '__main__':
    label_map = [''] * 90
    label_map[0] = 'person'
    yolo = Yolo(model_path='../IR/Yolo/coco', \
                device='GPU', \
                label_map=label_map, \
                img_width=768, \
                img_height=432)

    cap = cv2.VideoCapture('/root/sample-videos/store-aisle-detection.mp4')
    start_time = time.time()
    frame_count = 0
    fps = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        objects = yolo.inference(frame)
        yolo.draw_objects(frame, objects)

        cv2.putText(frame,
                    'FPS : ' + str(round(fps, 1)), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        time_gap = time.time() - start_time
        if time_gap > 1:
            fps = frame_count / time_gap
            start_time = time.time()
            frame_count = 0
            img_name = str(idx) + '.png'
            cv2.imwrite(os.path.join("./output", img_name), frame)
            idx += 1

        cv2.imshow("test", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        frame_count += 1
    cv2.destroyAllWindows()
    sys.exit(0)
