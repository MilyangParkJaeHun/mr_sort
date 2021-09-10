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

import cv2
import numpy as np
from openvino.inference_engine import IECore
import time

class Ssd_mobilenet:
  def __init__(self, model_path, device, edie_id, img_width, img_height, label_map, prob_threshold=0.5):
    self.model_xml = model_path+'/frozen_inference_graph.xml'
    self.model_bin = model_path+'/frozen_inference_graph.bin'
    print(self.model_xml)
    print(self.model_bin)

    self.device_name = device
    print('device  : ', self.device_name)

    self.edie_id = edie_id
    self.img_width = img_width
    self.img_height = img_height
    self.label_map = label_map
    print('edie id : ', label_map[edie_id])

    self.num_requests = 2
    self.cur_request_id = 0
    self.next_request_id = 1
    self.iou_threshold = 0.4
    self.prob_threshold = prob_threshold

    self.origin_im_size = (self.img_height, self.img_width)

    print("Reading IR...")
    self.ie = IECore()
    self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
    self.input_blob = next(iter(self.net.inputs))
    self.out_blob = next(iter(self.net.outputs))
    print('input  : ', self.input_blob)
    print('output : ', self.out_blob)
    print("Loading IR to the plugin...")
    self.ie.load_network(network=self.net, device_name=self.device_name, num_requests=self.num_requests)
    self.exec_net = self.ie.load_network(network=self.net, device_name=self.device_name, num_requests=self.num_requests)
    self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape

  def preprocess_frame(self, frame):
    in_frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
    return in_frame

  def filter_objects(self, objects, prob_threshold):
    return tuple(obj for obj in objects if(obj['confidence'] >= prob_threshold and \
                                           obj['class_id'] == self.edie_id))
#return tuple(obj for obj in objects if(obj['confidence'] >= prob_threshold))

  def validation_objects(self, objects):
    for obj in objects:
      obj['xmax'] = min(obj['xmax'], self.origin_im_size[1])
      obj['ymax'] = min(obj['ymax'], self.origin_im_size[0])
      obj['xmin'] = max(obj['xmin'], 0)
      obj['ymin'] = max(obj['ymin'], 0)

    return objects

  def get_objects(self, output, source_height_width):
    src_h, src_w = source_height_width
    objects = list()
    output = output.flatten()
    total_len = int(len(output)/7)

    for idx in range(total_len):
      base_index = idx * 7
      class_id = int(output[base_index + 1]) - 1
      prob = float(output[base_index + 2])
      xmin = int(output[base_index + 3] * src_w)
      ymin = int(output[base_index + 4] * src_h)
      xmax = int(output[base_index + 5] * src_w)
      ymax = int(output[base_index + 6] * src_h)

      now_object = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=prob)
      objects.append(now_object)

    return objects

  def switching_request_id(self):
    self.fcur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

  def inference(self, frame):
    objects = list()
    in_frame = self.preprocess_frame(frame)
    self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})
    if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
      output = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
      objects = self.get_objects(output, (self.img_height, self.img_width))
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
  # label_map = ['white','red','orange','yellow','green','sky','blue','mint','pink','purple','darkgreen','beige','brown','gray','black']
  label_map = [''] * 90
  label_map[0] = 'person'
  model = Ssd_mobilenet(model_path='../IR/Ssd/ssdlite', \
              device='GPU', \
              edie_id=0, \
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
      objects = model.inference(frame)
      model.draw_objects(frame, objects)

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