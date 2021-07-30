#!/bin/bash
docker run -it -d \
	-u openvino \
	-w /home/openvino/catkin_ws/src \
	--name mr_sort \
	--restart=unless-stopped \
	--device /dev/dri:/dev/dri \
	--device /dev/video0 \
	--device-cgroup-rule='c 189:* rmw' \
	--net=host -e DISPLAY \
	--gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v /tmp/.X11-unix \
	-v /dev/bus/usb:/dev/bus/usb \
	-v /home/park/dev/MOT/src/sort:/home/openvino/dev/sort \
	-v /home/park/dev/MOT/src/mr_sort:/home/openvino/dev/mr_sort \
	--ipc=host \
	--pid=host \
	mr_sort:latest
