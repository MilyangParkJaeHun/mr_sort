"""
Generate MOT GT data from labeling data collected by the Yolo_mark tool
Yolo mark data format : [obj_id, x_center, y_center, width, height]
MOT GT data format    : [frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]
"""
import os
import csv
import glob
import argparse

img_width = 640
img_height = 480

def parse_yolo(frame_id, line):
    global img_width, img_height
    data = line.split(" ")
    bbox = dict()

    x_center = float(data[1]) * img_width
    y_center = float(data[2]) * img_height
    width    = float(data[3]) * img_width
    height   = float(data[4]) * img_height

    bbox["frame"]       = frame_id
    bbox["id"]          = int(data[0])
    bbox["bb_left"]     = int(x_center - int(width/2))
    bbox["bb_top"]      = int(y_center - int(height/2))
    bbox["bb_width"]    = width
    bbox["bb_height"]   = height
    bbox["conf"]        = 1 # Always 1
    bbox["class"]       = 1 # Pedestrian class
    bbox["visibility"]  = 1

    return bbox

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Generate GT data")
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='../data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--yolo_path", help="Path to yolo data.", type=str, default='../data/yolo')

    args = parser.parse_args()
    return args

def selectBigger(a, b):
    a_width = a['bb_width']
    a_height = a['bb_height']
    a_area = a_width * a_height
    b_width = b['bb_width']
    b_height = b['bb_height']
    b_area = b_width * b_height
    if a_area > b_area:
        a['visibility'] = b_area / a_area * 100
        return a
    else:
        b['visibility'] = a_area / b_area * 100
        return b

if __name__ == "__main__":
    args = parse_args()
    seq_path = args.seq_path
    phase = args.phase
    yolo = args.yolo_path

    pattern = os.path.join(yolo, '*')
    for yolo_dir in glob.glob(pattern):
        seq = yolo_dir[pattern.find('*'):].split(os.path.sep)[0]
        print(seq)

        gt_fn = os.path.join(seq_path, phase, seq, 'gt')
        if not os.path.exists(gt_fn):
            os.makedirs(gt_fn)

        bbox_list = []
        with open(os.path.join(gt_fn, 'gt.txt'), 'w') as out_file:
            for yolo_fn in os.listdir(yolo_dir):
                print(os.path.join(yolo_dir, yolo_fn))
                with open(os.path.join(yolo_dir, yolo_fn), 'r') as in_file:
                    frame_id = int(yolo_fn.split('.')[0])

                    bbox_dict = dict()
                    while True:
                        line = in_file.readline()
                        if not line:
                            break

                        bbox = parse_yolo(frame_id, line)
                    
                        if bbox['id'] not in bbox_dict.keys():
                            bbox_dict[bbox['id']] = [bbox]
                        else:
                            bbox_dict[bbox['id']].append(bbox)
                    
                    for key in bbox_dict.keys():
                        if len(bbox_dict[key]) == 2:
                            bbox = selectBigger(bbox_dict[key][0], bbox_dict[key][1])
                            bbox_list.append(bbox)
                        else:
                            bbox_list.append(bbox_dict[key][0])
            
            bbox_list = sorted(bbox_list, key=(lambda x: [x['id'], x['frame']]))

            for bbox in bbox_list:
                print('%d,%d,%d,%d,%d,%d,%d,%d,%.2f'%(\
                    bbox['frame'], bbox['id'], bbox['bb_left'], bbox['bb_top'], bbox['bb_width'], bbox['bb_height'], \
                    bbox['conf'], bbox['class'], bbox['visibility']), file=out_file)
