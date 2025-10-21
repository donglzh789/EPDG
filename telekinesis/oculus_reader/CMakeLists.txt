cmake_minimum_required(VERSION 3.0.2)
project(oculus_reader)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  tf2_ros
)


catkin_python_setup()


catkin_package(
  CATKIN_DEPENDS
    rospy
    geometry_msgs
    tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/visualize_oculus_transforms.py
  scripts/reader.py
  scripts/FPS_counter.py
  scripts/buttons_parser.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/scripts
)

