# mr_sort
Simple, online, and realtime tracking for mobile robot

<p align="center">
<img src=https://user-images.githubusercontent.com/22341340/134169357-9407a12c-7a6b-45f6-988e-8f68d8908f91.gif width="640" height="240">
</p>

by Park JaeHun

## Introduction
Multiple object tracking suitable for mobile robots using the two methods below.
1. Calibration camera rotation using wheel encoder based odometry information
2. Dynamically set the tracker's life period

## Environments
- Ubuntu 20.04
- ROS noetic
- OpenVINO 2021.3.394
- OpenCV 4.5.2-openvino
- Numpy 1.17.3

## Install
```
$ git clone https://github.com/MilyangParkJaeHun/mr_sort.git
$ cd mr_sort
$ git submodule init
$ git submodule update
```

## Demo Run
1. Downloads PxRx sequence dataset
```
$ cd /path/to/mr_sort/data
$ source source data_downloads.sh
```
2. Run
```
$ python3 mr_sort.py --seq_path=data --display
```

## Using different detection model quantized by [OpenVINO](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- A example of using SSDLite-MobileNet model
  ```
  $ cd /path/to/mr_sort/openvino_detector/IR/Ssd
  $ source model_downloads.sh
  $ cd /path/to/mr_sort/openvino_detector
  $ python3 mot_detector.py --model_type=ssd \
                            --model_path=IR/Ssd/ssdlite_coco \
                            --img_path=../tracker/data \
                            --device=CPU --display
  ```
For more information, see [here](https://github.com/MilyangParkJaeHun/openvino_detector)

## Using in your own project
Below is the gist of how to use mr_sort. See the ['main'](https://github.com/MilyangParkJaeHun/mr_sort/blob/fd0adc0b6b2ad8c55c98e6d8ab20570c99791093/tracker/mr_sort.py#L353) section of [mr_sort.py](https://github.com/MilyangParkJaeHun/mr_sort/blob/fd0adc0b6b2ad8c55c98e6d8ab20570c99791093/tracker/mr_sort.py) for a complete example.
```
from mr_sort import *

#create instance of MR_SORT
mot_tracker = Mrsort() 

# get detections
...

# get odometry
# odometry is [th(degree), x, y] format array
...

# update MR_SORT
track_bbs_ids = mot_tracker.update(detections, odometry)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...

```
## Reference
SORT : https://github.com/abewley/sort
