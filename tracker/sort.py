import os
import glob
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from scipy.optimize import linear_sum_assignment


from utils import KalmanFilter

from typing import List

np.random.seed(0)

def x_to_bbox(x):
    u, v, s, r = x[:4]
    w = np.sqrt(s * r)
    h = s / w
    
    xmin = u - w/2.
    ymin = v - h/2.
    xmax = u + w/2.
    ymax = v + h/2.
    return np.array([xmin, ymin, xmax, ymax])

def bbox_to_z(bbox):
    xmin, ymin, xmax, ymax, _ = bbox
    w = xmax - xmin
    h = ymax - ymin
    u = xmin + w/2.
    v = ymin + h/2.
    s = w * h
    r = w / float(h)
    return np.array([u, v, s, r]).reshape((4, 1))

class BoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.id = BoxTracker.count
        BoxTracker.count += 1

        # x = [u, v, s, r, u', v', s']
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]])
        self.kf.P *= 10.
        self.kf.P[4:,4:] *= 100. # 초기 속도 예측에 대한 높은 불확실성을 반영
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q[-1,-1] *= 0.01
        self.kf.R[2:,2:] *= 10.

        self.kf.x[:4] = bbox_to_z(bbox)

        self.age = 0
        self.hits = 0
        self.time_since_update = 0

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        next_bbox = x_to_bbox(self.kf.x)

        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1

        return next_bbox

    def update(self, bbox):
        self.kf.update(bbox_to_z(bbox))
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        return x_to_bbox(self.kf.x)
    
    def get_scale(self):
        return self.kf.x[2]

class Sort(object):
    def __init__(self, t_lost=1, t_probation=3, iou_threshold=0.3):
        self.t_lost = t_lost
        self.t_probation = t_probation
        self.iou_threshold = iou_threshold

        self.trackers: List[BoxTracker] = []

        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1

        def notValidPred(scale, bbox):
            return scale <= 0 or np.any(np.isnan(bbox))

        preds = []
        error_preds = []
        for i, trk in enumerate(self.trackers): # 추적 예측 단계
            pred_bbox = trk.predict()
            s = trk.get_scale()

            if notValidPred(s, pred_bbox):
                error_preds.append(i)
            else:
                preds.append(pred_bbox)
        preds = np.array(preds).reshape(-1,4)

        for i in reversed(error_preds):
            self.trackers.pop(i)
        
        matches, unmatched_dets, unmatched_trks = self.match_dets_with_trks(dets, preds) # 검출 결과와 추적 예측 결과 매칭

        for m in matches: # 매칭에 성공한 검출 결과를 기반으로 추적기 업데이트
            self.trackers[m[1]].update(dets[m[0]])

        for i in unmatched_dets: # 매칭에 실패한 검출 결과를 추적하는 새로운 추적기 생성
            trk = BoxTracker(dets[i])
            self.trackers.append(trk)

        trk_res = []
        for i, trk in reversed(list(enumerate(self.trackers))):
            bbox = trk.get_state()
            if trk.hits >= self.t_probation or self.frame_count <= self.t_probation: # 추적을 t_probation 동안 유지한 경우만 결과 사용
                trk_res.append([*bbox, trk.id + 1])
            if trk.time_since_update > self.t_lost: # t_lost 동안 추적에 실패하면 추적 중단
                self.trackers.pop(i)
        return np.array(trk_res, dtype=object)

    def match_dets_with_trks(self, dets, preds):
        def cal_iou(bb_a, bb_b): # iou score 계산
            bb_b = np.expand_dims(bb_b, 0)
            bb_a = np.expand_dims(bb_a, 1)

            xmin_batch = np.maximum(bb_a[..., 0], bb_b[..., 0])
            ymin_batch = np.maximum(bb_a[..., 1], bb_b[..., 1])
            xmax_batch = np.minimum(bb_a[..., 2], bb_b[..., 2])
            ymax_batch = np.minimum(bb_a[..., 3], bb_b[..., 3])

            w_batch = np.maximum(0., xmax_batch - xmin_batch)
            h_batch = np.maximum(0., ymax_batch - ymin_batch)
            overlap = w_batch * h_batch
            union = (bb_a[..., 2] - bb_a[..., 0]) * (bb_a[..., 3] - bb_a[..., 1]) + \
                    (bb_b[..., 2] - bb_b[..., 0]) * (bb_b[..., 3] - bb_b[..., 1]) - overlap
            
            iou_matrix = overlap / union

            return iou_matrix

        iou_matrix = cal_iou(dets, preds)

        matches = []
        unmatched_dets = []
        unmatched_trks = []
        if min(iou_matrix.shape) > 0: # 매칭의 가능성이 있다면 iou score 기반으로 최적 매칭 서치
            det_indices, pred_indices = linear_sum_assignment(-iou_matrix) # 헝가리안 알고리즘
            matched_indices = np.array(list(zip(det_indices, pred_indices)))
            
            for det_i, pred_i in matched_indices:
                if(iou_matrix[det_i][pred_i] < self.iou_threshold):
                    unmatched_dets.append(det_i)
                    unmatched_trks.append(pred_i)
                else:
                    matches.append([det_i, pred_i])

            for i in range(len(dets)):
                if i not in matched_indices[:, 0]:
                    unmatched_dets.append(i)

            for i in range(len(preds)):
                if i not in matched_indices[:, 1]:
                    unmatched_trks.append(i)
            
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(preds)))
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.array(matches)
        
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) #used only for display
    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt') 
    for seq_dets_fn in glob.glob(pattern):
        BoxTracker.count = 0
        motracker = Sort(t_lost=args.max_age, 
                        t_probation=args.min_hits,
                        iou_threshold=args.iou_threshold)
        
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
            print("Processing %s."%(seq))
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1 #detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if(display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
                    im =io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')
                
                start_time = time.time()
                trackers = motracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                
                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
                    
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    if(display):
        print("Note: to get real runtime results run without the option: --display")