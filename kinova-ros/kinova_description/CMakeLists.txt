cmake_minimum_required(VERSION 3.0.2)
project(kinova_description)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

install(DIRECTORY urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY assembly DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY LEAP_Hand_Sim DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY meshes
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
   FILES_MATCHING PATTERN "*.dae"
 )

install(FILES
  launch/system_display.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)