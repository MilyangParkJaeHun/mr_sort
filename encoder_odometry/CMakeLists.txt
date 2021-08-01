cmake_minimum_required(VERSION 3.0.2)
project(encoder_odometry)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  std_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES encoder_odometry
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp std_msgs tf
  DEPENDS system_lib
)

include_directories(
#  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(encoder_odometry_node src/odometry.cpp)

target_link_libraries(encoder_odometry_node ${catkin_LIBRARIES})

add_dependencies(encoder_odometry_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})