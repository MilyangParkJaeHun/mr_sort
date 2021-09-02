from DetModel import OpenvinoDet

import cv2
import time
import os
import numpy as np
from math import exp as exp
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

class Yolov4(OpenvinoDet):

    def __init__(self, model_path, device, label_map, prob_threshold=0.5):
        """
        Set key parameters for Detector
        """
        self.model_path = model_path
        self.device = device
        self.label_map = label_map
        self.prob_threshold = prob_threshold
        self.iou_threshold = 0.4

        self.img_height = 480
        self.img_width  = 640
        self.before_frame = np.empty((self.img_height, self.img_width, 3))
        self.model_xml = model_path+'/frozen_darknet_yolov4_model.xml'
        self.model_bin = model_path+'/frozen_darknet_yolov4_model.bin'
        print(self.model_xml)
        print(self.model_bin)

        self.num_requests = 2
        self.cur_request_id = 0
        self.next_request_id = 1

        print("Reading IR...")
        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        print('input  : ', self.input_blob)
        print('output : ', self.out_blob)
        print("Loading IR to the plugin...")
        self.ie.load_network(
            network=self.net, device_name=self.device, num_requests=self.num_requests)
        self.exec_net = self.ie.load_network(
            network=self.net, device_name=self.device, num_requests=self.num_requests)
        self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape

    def inference(self, frame):
        self.img_height = frame.shape[0]
        self.img_width = frame.shape[1]
        in_frame = self.preprocess_frame(frame)

        self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})
        dets = list()
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            output = self.exec_net.requests[self.cur_request_id].output_blobs
            dets = self.parse_output(output, self.net, (self.h, self.w), (self.img_height, self.img_width), self.prob_threshold)
            dets = self.filter_dets(dets)
            dets = self.validation_dets(dets)
        self.switching_request_id()
        self.update_frame(frame)
        return dets

    def clear(self):
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)
        print("Successfully reload!!!")

    def preprocess_frame(self, frame):
        return super().preprocess_frame(frame, self.n, self.c, self.h, self.w)

    def filter_dets(self, dets):
        return super().filter_dets(dets, self.prob_threshold, self.iou_threshold)
    
    def validation_dets(self, dets):
        return super().validation_dets(dets, self.img_height, self.img_width)

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
        dets = list()
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
            dets.append(self.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence, \
                                        im_h=orig_im_h, im_w=orig_im_w))
        return dets
    
    def parse_output(self, output, net, new_frame_height_width, source_height_width, prob_threshold):
        dets = list()
        function = ng.function_from_cnn(net)
        for layer_name, out_blob in output.items():
            out_blob = out_blob.buffer.reshape(net.outputs[layer_name].shape)
            params = [x._get_attributes() for x in function.get_ordered_ops() if x.get_friendly_name() == layer_name][0]
            layer_params = YoloParams(params, out_blob.shape[2])
            dets += self.parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                        prob_threshold)
        return dets

    def switching_request_id(self):
        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

if __name__ == "__main__":
    label_map = [''] * 90
    label_map[0] = 'person'
    yolo = Yolov4(model_path='IR/Yolo/coco', \
                device='GPU', \
                label_map=label_map)

    cap = cv2.VideoCapture('/home/openvino/sample-videos/store-aisle-detection.mp4')
    start_time = time.time()
    frame_count = 0
    fps = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = yolo.inference(frame)
        out_frame = yolo.get_results_img(dets)

        cv2.putText(out_frame,
                    'FPS : ' + str(round(fps, 1)), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        time_gap = time.time() - start_time
        if time_gap > 1:
            fps = frame_count / time_gap
            start_time = time.time()
            frame_count = 0
            img_name = str(idx) + '.png'
            cv2.imwrite(os.path.join("./det_output", img_name), out_frame)
            idx += 1

        cv2.imshow("test", out_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        frame_count += 1
    cv2.destroyAllWindows()
