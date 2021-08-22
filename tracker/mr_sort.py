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
import configparser
from kalman import KalmanFilter
import argparse
import time
import glob
from skimage import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import os
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# from filterpy.kalman import KalmanFilter


config = configparser.ConfigParser()
np.random.seed(0)

fov = 140  # camera's field of view
PI = 3.14159265
frame_width = 640
frame_height = 480
min_height = 50

pimg_w = 640
pimg_h = 480

pbbox_height = 100
pbbox_width = 100


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



def convert_polar_to_bbox(p, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    global fov, frame_width, frame_height

    theta = p[0]
    r = p[1]
    ar = p[3]

    x = float(theta * frame_width / fov)
    y = p[2]
    h = frame_height - (r - min_height) * frame_height / (frame_height - min_height)
    w = ar * h

    if(score == None):
        return np.array([x-w/2., y-h/2., x+w/2., y+h/2.]).reshape((1, 4))
    else:
        return np.array([x-w/2., y-h/2., x+w/2., y+h/2., score]).reshape((1, 5))

def convert_bbox_to_polar(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    global fov, frame_width, frame_height

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    ar = w / h

    theta = (x / frame_width) * fov
    r = min_height + (frame_height - min_height) * (frame_height - h) / frame_height

    return np.array([theta, r, y, ar]).reshape((4, 1))

def convert_polar_to_pbbox(polar):
    global pbbox_height, pbbox_width, frame_width, frame_height

    theta = polar[0]
    r = polar[1]

    w = pbbox_width
    h = pbbox_height

    x = int(frame_width / 2 - r *
            math.cos((PI - degree_to_rad(fov))/2 + degree_to_rad(theta)))
    y = int(frame_height - r *
            math.sin((PI - degree_to_rad(fov))/2 + degree_to_rad(theta)))

    return np.array([x - int(w/2), y - int(h/2), x + int(w/2), y + int(h/2)]).reshape((1, 4))

def convert_bbox_to_pbbox(bbox):
    global pbbox_height

    p = convert_bbox_to_polar(bbox)
    pbbox = convert_polar_to_pbbox(p)[0]

    if len(bbox) == 5:
        return np.array([pbbox[0], pbbox[1], pbbox[2], pbbox[3], bbox[4]])
    else:
        return np.array(pbbox)



def draw_polar_coordinate(frame, d, color, frame_w, frame_h):
    cv_color = color * 255
    cv_color = [cv_color[2], cv_color[1], cv_color[0]]
    global frame_width, frame_height

    # bbox = d[:4]
    # bbox = bbox

    # polar = convert_bbox_to_polar(bbox)
    # pbbox = fit_polar_frame(convert_polar_to_pbbox(polar)[0])
    pbbox = fit_polar_frame(convert_bbox_to_pbbox(d))

    x = int((pbbox[0] + pbbox[2])/2)
    y = int((pbbox[1] + pbbox[3])/2)
    w = (pbbox[2] - pbbox[0])
    h = (pbbox[3] - pbbox[1])

    frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (x - int(w/2), y - int(h/2)),
                          (x + int(w/2), y + int(h/2)), cv_color, 2)

    return frame


class KalmanPolarBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
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

        self.kf.x[:4] = convert_bbox_to_polar(bbox)
        self.time_since_update = 0
        self.id = KalmanPolarBoxTracker.count
        KalmanPolarBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # bbox -> polar
        self.kf.update(convert_bbox_to_polar(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # polar -> pbbox
        self.history.append(convert_polar_to_pbbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # polar -> bbox
        return convert_polar_to_bbox(self.kf.x)


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

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # pos format : pbbox
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # if len(dets) == 0:
        #     pdets = dets
        # else:
        #     pdets = np.squeeze(
        pdets = np.array([convert_bbox_to_pbbox(det) for det in dets]).reshape(-1, 5)

        # dets, trks foramt : pbbox
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            pdets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            # dets format : bbox
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanPolarBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # polar -> bbox
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

def fit_polar_frame(bbox):
    global frame_width, frame_height, pimg_w, pimg_h

    xmin = (bbox[0]/frame_width)*pimg_w
    ymin = (bbox[1]/frame_height)*pimg_h
    xmax = (bbox[2]/frame_width)*pimg_w
    ymax = (bbox[3]/frame_height)*pimg_h

    return np.array([xmin, ymin, xmax, ymax]).astype(np.int32)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--seq_path", 
                        help="Path to detections.", type=str, default='data')
    parser.add_argument("--out_path", 
                        help="Path to output.", type=str, default='mr_output')
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase

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


    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    print(pattern)

    # Draw polar base frame
    white = (255, 255, 255)
    green = (100, 255, 100)

    polar_img = np.zeros((int(pimg_h*1.2), pimg_w, 3), np.uint8)
    intercept = pimg_h - pimg_w/2*math.tan((PI-degree_to_rad(fov))/2)
    if intercept > 0:
        polar_img = cv2.line(
            polar_img, (int(pimg_w/2), pimg_h), (0, int(intercept)), white, 1)
        polar_img = cv2.line(polar_img, (int(pimg_w/2), pimg_h),
                            (pimg_w, int(intercept)), white, 1)
    else:
        end1 = (-1*intercept) / math.tan((PI-degree_to_rad(fov))/2)
        end2 = pimg_w - end1
        polar_img = cv2.line(
            polar_img, (int(pimg_w/2), pimg_h), (end1, pimg_h), white, 1)
        polar_frame = cv2.line(
            polar_img, (int(pimg_w/2), pimg_h), (end2, pimg_h), white, 1)
    polar_img = cv2.circle(polar_img, (int(pimg_w/2), pimg_h), 10, green, 20)

    # sequnce pattern loop
    # pbbox_shape_list = [[a, b] for a in range(5, 25, 5) for b in range(5, 25, 5)]
    pbbox_shape_list = [[20, 10]]

    for pbbox_shape in pbbox_shape_list:
        print('pbbox shape : ', pbbox_shape)

        output_path = os.path.join('static', '%d_%d'%(pbbox_shape[0], pbbox_shape[1]))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

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

            pbbox_width = int(frame_width * pbbox_shape[0] / 100)
            pbbox_height = int(frame_height * pbbox_shape[1] / 100)

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
                    polar_frame = polar_img.copy()

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

                    start_time = time.time()
                    trackers = mot_tracker.update(dets)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    for d in dets:
                        pbbox = fit_polar_frame(convert_bbox_to_pbbox(d))
                        polar_frame = cv2.rectangle(polar_frame, (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]), (255, 255, 255), 2)


                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame,
                            d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]), file=out_file)
                        if(display):
                            d = d.astype(np.int32)
                            ax1.add_patch(patches.Rectangle(
                                (d[0], d[1]), d[2]-d[0], d[3]-d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                            polar_frame = draw_polar_coordinate(
                                polar_frame, d, colours[d[4] % 32, :], pimg_w, pimg_h)

                    if(display):
                        fig.canvas.flush_events()
                        plt.draw()
                        plt.savefig(os.path.join(
                            output_path, seq, '%06d.jpg' % (frame)))

                        ax1.cla()

                        cv2.imshow('polar', polar_frame)
                        cv2.imwrite(os.path.join(output_path, seq,
                                    'polar_%06d.jpg' % (frame)), polar_frame)

                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            break

        print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
            (total_time, total_frames, total_frames / total_time))

        if(display):
            print("Note: to get real runtime results run without the option: --display")
