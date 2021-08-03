cmake_minimum_required(VERSION 3.0.2)
project(data_collector)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES data_collector
  CATKIN_DEPENDS geometry_msgs nav_msgs rospy std_msgs sensor_msgs
  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/collector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)