"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from filterpy.kalman import EKF
import configparser

from numpy import ma
from kalman import KalmanFilter
import argparse
import time
import glob
from skimage import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import os
import cv2
import sys
import math
import numpy as np
import matplotlib
from functools import cmp_to_key
matplotlib.use('TkAgg')


config = configparser.ConfigParser()
np.random.seed(0)

fov = 113  # camera's field of view
PI = 3.14159265
frame_width = 640
frame_height = 480
min_height = 50

pimg_w = 640
pimg_h = 480

pbbox_height = 100
pbbox_width = 100

min_ratio = 0.05
max_ratio = 0.3


def degree_to_rad(degree):
    return degree / 180 * PI


def rad_to_degree(rad):
    return rad / PI * 180


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def convert_x_to_th(x):
    global frame_width, fov

    return x / frame_width * fov

def convert_th_to_x(th):
    global frame_width, fov

    return th * frame_width / fov

def cal_cam_move(x, th):
    x[0] += convert_th_to_x(th)

    return x

def h_compare(x, y):
    if x[0] > y[0]:
        return 1
    elif x[0] == y[0]:
        if x[1] < y[1]:
            return 1
        else:
            return 0
    else:
        return 0

class PedestrianMask(object):
    def __init__(self):
        self.max_age_bound = 20
        self.area_mask = []
        self.dx_mask = []
        self.mask_list = []
        self.mask_count = 0

    def clear(self):
        self.area_mask.clear()
        self.dx_mask.clear()
        self.mask_list.clear()
        self.mask_count = 0

    def update(self):
        sorted(self.mask_list, key=cmp_to_key(h_compare))

        for mask in self.mask_list:
            self.mask_count +=1
            self.area_mask.append(mask[1:3])
            self.dx_mask.append(mask[3])

    def append(self, bbox, dx):
        xmin = bbox[0]
        xmax = bbox[2]
        h = bbox[3] - bbox[1]
        self.mask_list.append([h, xmin, xmax, dx])

    def get_escape_time(self, bbox, dx):
        # print(self.area_mask[x])
        xmin = bbox[0]
        xmax = bbox[2]

        escape_time = -1
        for i in range(self.mask_count):
            area = self.area_mask[i]
            mask_dx = self.dx_mask[i]
            dx -= mask_dx
            if (xmin <= area[1] and xmin >= area[0]) or (xmax <= area[1] and xmax >= area[0]):
                if dx > 0:
                    escape_time = (area[1] - xmin) / dx
                    break
                elif dx < 0:
                    escape_time = (xmax - area[0]) / dx
                    break
                else:
                    escape_time = self.max_age_bound
                    break
        if escape_time > self.max_age_bound:
            escape_time = self.max_age_bound
        
        return escape_time

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, max_age):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.dx = self.kf.x[4]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.is_occluded = False
        self.default_max_age = max_age
        self.max_age = self.default_max_age

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.max_age = self.default_max_age
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.dx = self.kf.x[4]
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.mask = PedestrianMask()

    def update(self, dets=np.empty((0, 5)), odom=[0, 0, 0]):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        self.mask.clear()
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # pos format : pbbox
            d_th = odom[0]
            # print('before : ', self.trackers[t].kf.x[0])
            cal_cam_move(self.trackers[t].kf.x, d_th)
            # print('after : ', self.trackers[t].kf.x[0])
            # print('----------------------------------------------------')
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].id+1]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

            dx = self.trackers[m[1]].kf.x[4]
            self.mask.append(dets[m[0]], dx)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], self.max_age)
            self.trackers.append(trk)

            self.mask.append(dets[i], 0)

        if len(unmatched_trks) > 0:
            self.mask.update()

        # print('----------------------------------------')
        for i in unmatched_trks:
            bbox = convert_x_to_bbox(self.trackers[i].kf.x)[0]
            dx = float(self.trackers[i].kf.x[4])
            max_age = self.mask.get_escape_time(bbox, dx)
            # print(max_age)

            if max_age == -1:                
                self.trackers[i].is_occluded = False
                self.trackers[i].max_age = max_age
            else:
                self.trackers[i].is_occluded = True
                self.trackers[i].max_age = self.max_age

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)) or trk.is_occluded:
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            max_age = self.max_age
            if trk.is_occluded:
                max_age = trk.max_age

            if(trk.time_since_update > max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return (np.concatenate(ret), trks)
        return (np.empty((0, 5)), trks)


def read_odom(odom_fn):
    before_th = 0
    before_x = 0
    before_y = 0

    info = dict()
    with open(odom_fn, 'r') as in_file:
        start = True
        while True:
            line = in_file.readline()
            if not line:
                break
            data = line[:-1].split(',')
            img_fn = data[0]
            th = float(data[1])
            x = float(data[2])
            y = float(data[3])

            if start:
                info[img_fn] = [0, 0, 0]
                start = False
            else:
                info[img_fn] = [rad_to_degree(th-before_th), x-before_x, y-before_y]

            before_th = th
            before_x = x
            before_y = y

    return info


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--seq_path",
                        help="Path to detections.", type=str, default='data')
    parser.add_argument("--out_path",
                        help="Path to output.", type=str, default='odom_output')
    parser.add_argument("--phase",
                        help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--min_ratio",
                        help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--max_ratio",
                        help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    min_ratio = args.min_ratio
    max_ratio = args.max_ratio

    data_path = args.seq_path
    output_path = args.out_path

    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display

    if(display):
        if not os.path.exists(data_path):
            print('\n\tERROR: data not found!\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    print(pattern)

    # sequnce pattern loop
    for seq_dets_fn in glob.glob(pattern):

        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        # read sequnce info
        info_fn = os.path.join(data_path, phase, seq, 'seqinfo.ini')
        if not os.path.exists(info_fn):
            print('\n\tERROR: sequcne info not found!\n\n')
            exit()
        config.read(info_fn)
        frame_width = int(config['Sequence']['imWidth'])
        frame_height = int(config['Sequence']['imHeight'])

        # read odom info
        odom_fn = os.path.join(data_path, phase, seq, 'odom', 'odom.txt')
        odom_info = dict()
        odom_mode = False
        if not os.path.exists(odom_fn):
            print("Run non odom mode : odom file isn't exists")

        else:
            odom_info = read_odom(odom_fn)
            odom_mode = True

        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker

        if display and not os.path.exists(os.path.join(output_path, seq)):
            os.makedirs(os.path.join(output_path, seq))

        with open(os.path.join(output_path, '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            print("Image width  : ", frame_width)
            print("Image height : ", frame_height)

            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                img_fn = '%06d.jpg' % (frame)
                if odom_mode:
                    odom = odom_info[img_fn]
                else:
                    odom = [0, 0, 0]
                det_frame = cv2.imread(os.path.join(
                    data_path, phase, seq, 'img1', img_fn))
                odom_frame = cv2.imread(os.path.join(
                    data_path, phase, seq, 'img1', img_fn))

                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if(display):
                    fn = os.path.join(data_path, phase, seq,
                                      'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                    odom_frame = cv2.line(odom_frame, (int(
                        pimg_w/2), int(pimg_h*0.9)), (int(pimg_w/2), int(pimg_h*0.9)), (255, 0, 0), 5)
                    arrow_end = (
                        int(pimg_w/2 - 10*(odom[0]/180)*pimg_w/2), int(pimg_h*0.9))
                    if arrow_end[0] < 0:
                        arrow_end = list(arrow_end)
                        arrow_end[0] = 1
                        arrow_end = tuple(arrow_end)
                    if arrow_end[0] > pimg_w:
                        arrow_end = list(arrow_end)
                        arrow_end[0] = pimg_w
                        arrow_end = tuple(arrow_end)
                    arrow_color = (255, 0, 0)
                    if arrow_end[0] > pimg_w/2:
                        arrow_color = (0, 255, 0)
                    else:
                        arrow_color = (0, 0, 255)

                    if odom_mode:
                        odom_frame = cv2.arrowedLine(
                            odom_frame, (int(pimg_w/2), int(pimg_h*0.9)), arrow_end, arrow_color, 5)
                        odom_frame = cv2.putText(odom_frame, 'degree : %.2f' % (odom[0]), (30, int(
                            pimg_h*0.8)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)

                start_time = time.time()
                trackers, predicts = mot_tracker.update(dets, odom)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                if display:
                    mask_list = mot_tracker.mask.area_mask
                    for d in dets:
                        d = d.astype(np.int32)
                        det_frame = cv2.rectangle(
                            det_frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 0), 2)
                        odom_frame = cv2.rectangle(
                            odom_frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 0), 2)

                        for mask in mask_list:
                            xmin = int(mask[0])
                            xmax = int(mask[1])
                            print(mask)
                            cv2.rectangle(odom_frame, (xmin, frame_height-50), (xmax, frame_height), (0, 0, 0), 2)


                    for id, pbbox in enumerate(predicts):
                        pbbox = pbbox.astype(np.uint32)
                        color = colours[pbbox[4] % 32, :]
                        cv_color = color * 255
                        cv_color = [cv_color[2], cv_color[1], cv_color[0]]
                        # print(pbbox)

                        if pbbox[0] < 0 or pbbox[0] > frame_width + 10:
                            pbbox[0] = 0
                        if pbbox[1] < 0 or pbbox[1] > frame_width + 10:
                            pbbox[1] = 0
                        # if pbbox[0] > frame_width:
                        #     pbbox[0] = frame_width
                        # if pbbox[1] > frame_height:
                        #     pbbox[1] = frame_height
                        if pbbox[2] > frame_width:
                            pbbox[2] = frame_width
                        if pbbox[3] > frame_height:
                            pbbox[3] = frame_height 

                        odom_frame = cv2.rectangle(
                            odom_frame, (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]), cv_color, 2)

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame,
                                                                    d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]), file=out_file)
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2]-d[0], d[3]-d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                        # odom_frame = draw_polar_coordinate(
                        # odom_frame, d, colours[d[4] % 32, :], pimg_w, pimg_h)

                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    plt.savefig(os.path.join(
                        output_path, seq, '%06d.jpg' % (frame)))

                    ax1.cla()

                    cv2.imshow('odom', odom_frame)
                    cv2.imshow('det', det_frame)
                    cv2.imwrite(os.path.join(output_path, seq,
                                'odom_%06d.jpg' % (frame)), odom_frame)
                    cv2.imwrite(os.path.join(output_path, seq,
                                'det_%06d.jpg' % (frame)), det_frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

    if total_time == 0:
        total_time = 1.
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))

    if(display):
        print("Note: to get real runtime results run without the option: --display")
