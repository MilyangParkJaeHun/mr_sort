"""
    mr_sort.py
    Author: Park Jaehun

    MR_SORT : A Simple, Online and Realtime Tracker for Mobile Robot
    Purpose
        Multiple object tracking suitable for mobile robots using the two methods below
        1. Calibration camera rotation using wheel encoder based odometry information
        2. Dynamically set the tracker's life period
"""

from __future__ import print_function

import os
import time
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage import io

import configparser
import argparse
from kalman import KalmanFilter

np.random.seed(0)

config = configparser.ConfigParser()
fov = 113  # camera's field of view
PI = 3.14159265
frame_width = 640
frame_height = 480

img_w = 640
img_h = 480

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
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
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
            self.hit_streak = int(self.hits / 10) - self.time_since_update + 2
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]

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

class Mrsort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), odom=[0, 0, 0]):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          odom - array of odometry info in the format [th, x, y]
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
            d_th = odom[0]
            cal_cam_move(self.trackers[t].kf.x, d_th)

            pos = self.trackers[t].predict()
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

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], self.max_age)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)) or trk.is_occluded:
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1

            # dynamically set the tracker's life period
            if(trk.time_since_update > self.max_age + trk.hits/10):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    print(pattern)

    # sequnce pattern loop
    for seq_dets_fn in glob.glob(pattern):
        KalmanBoxTracker.count = 0

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

        mot_tracker = Mrsort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the MRSORT tracker

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

                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if(display):
                    fn = os.path.join(data_path, phase, seq,
                                      'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' on MR_SORT')

                    arrow_end = (
                        int(img_w/2 - 10*(odom[0]/180)*img_w/2), int(img_h*0.98))
                    if arrow_end[0] < 0:
                        arrow_end = list(arrow_end)
                        arrow_end[0] = 1
                        arrow_end = tuple(arrow_end)
                    if arrow_end[0] > img_w:
                        arrow_end = list(arrow_end)
                        arrow_end[0] = img_w
                        arrow_end = tuple(arrow_end)
                    arrow_color = (1, 0, 0)
                    if arrow_end[0] > img_w/2:
                        arrow_color = (1, 0, 0)
                    else:
                        arrow_color = (0, 1, 0)

                    ax1.annotate("", xy=arrow_end, xytext=(int(img_w/2), int(img_h*0.98)),  arrowprops=dict(arrowstyle="->", color=arrow_color), size=20)

                start_time = time.time()
                trackers = mot_tracker.update(dets, odom)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame,
                                                                    d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]), file=out_file)
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2]-d[0], d[3]-d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    plt.savefig(os.path.join(
                        output_path, seq, '%06d.jpg' % (frame)))
                    ax1.cla()

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

    if total_time == 0:
        total_time = 1.
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))

    if(display):
        print("Note: to get real runtime results run without the option: --display")