from DetModel import OpenvinoDet

import cv2
import time
import os
import numpy as np
from openvino.inference_engine import IECore


class SsdMobilenet(OpenvinoDet):

    def __init__(self, model_path, device, label_map, prob_threshold=0.5):
        """
        Set key parameters for Detector
        """
        self.model_path = model_path
        self.device = device
        self.label_map = label_map
        self.prob_threshold = prob_threshold

        self.img_height = 480
        self.img_width  = 640
        self.before_frame = np.empty((self.img_height, self.img_width, 3))
        self.model_xml = model_path+'/frozen_inference_graph.xml'
        self.model_bin = model_path+'/frozen_inference_graph.bin'

        self.num_requests = 2
        self.cur_request_id = 0
        self.next_request_id = 1
        self.iou_threshold = 0.4

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
            output = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
            dets = self.parse_output(output, (self.img_height, self.img_width))
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

    def intersection_over_union(self, box_1, box_2):
        return super().intersection_over_union(box_1, box_2)

    def filter_dets(self, dets):
        return super().filter_dets(dets, self.prob_threshold, self.iou_threshold)

    def validation_dets(self, dets):
        return super().validation_dets(dets, self.img_height, self.img_width)
    
    def parse_output(self, output, source_height_width):
        src_h, src_w = source_height_width
        dets = list()
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

            det = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=prob)
            dets.append(det)

        return dets

    def switching_request_id(self):
        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id


if __name__ == "__main__":
    label_map = [''] * 90
    label_map[0] = 'person'
    model = SsdMobilenet(model_path='IR/Ssd/ssdlite_coco', \
                device='GPU',
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
        dets = model.inference(frame)
        out_frame = model.get_results_img(dets)

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