"""
    fiji2gt.py
    Author: Park Jaehun

    Purpose
        Generate MOT GT data from labeling data collected by the Fiji tool
        Fiji data format    : [id visibility 0.0 0 0.0 xmim ymin xmax ymax 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
        MOT GT data format  : [frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]
"""
import os
import csv
import glob
import argparse

def parse_fiji(frame_id, line):
    data = line.split(" ")
    bbox = dict()

    bbox["frame"]       = frame_id
    bbox["id"]          = int(data[0])
    bbox["bb_left"]     = int(data[5])
    bbox["bb_top"]      = int(data[6])
    bbox["bb_width"]    = int(data[7]) - int(data[5])
    bbox["bb_height"]   = int(data[8]) - int(data[6])
    bbox["conf"]        = 1 # Always 1
    bbox["class"]       = 1 # Pedestrian class
    bbox["visibility"]  = float(data[1]) / 100

    return bbox

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Generate GT data")
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='../data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--fiji_path", help="Path to Fiji data.", type=str, default='../data/fiji')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    seq_path = args.seq_path
    phase = args.phase
    fiji = args.fiji_path

    pattern = os.path.join(fiji, '*')
    for fiji_dir in glob.glob(pattern):
        seq = fiji_dir[pattern.find('*'):].split(os.path.sep)[0]
        print(seq)

        gt_fn = os.path.join(seq_path, phase, seq, 'gt')
        if not os.path.exists(gt_fn):
            os.makedirs(gt_fn)

        bbox_list = []
        with open(os.path.join(gt_fn, 'gt.txt'), 'w') as out_file:
            for fiji_fn in os.listdir(fiji_dir):
                print(os.path.join(fiji_dir, fiji_fn))
                with open(os.path.join(fiji_dir, fiji_fn), 'r') as in_file:
                    frame_id = int(fiji_fn.split('.')[0])

                    while True:
                        line = in_file.readline()
                        if not line:
                            break

                        bbox = parse_fiji(frame_id, line)
                        bbox_list.append(bbox)
            
            bbox_list = sorted(bbox_list, key=(lambda x: [x['id'], x['frame']]))

            for bbox in bbox_list:
                print('%d,%d,%d,%d,%d,%d,%d,%d,%.2f'%(\
                    bbox['frame'], bbox['id'], bbox['bb_left'], bbox['bb_top'], bbox['bb_width'], bbox['bb_height'], \
                    bbox['conf'], bbox['class'], bbox['visibility']), file=out_file)
