cmake_minimum_required(VERSION 3.0.2)
project(exploration)

## Compile as C++11, supported in ROS Noetic and newer
add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  message_generation

)
add_service_files(
  FILES
  AstarPath.srv
)

generate_messages(
  DEPENDENCIES
  nav_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs   message_runtime

)

catkin_install_python(PROGRAMS
  scripts/360_clip.py
  scripts/filter_scan.py
  scripts/frontier_mapping.py
  scripts/object_navigation.py
  scripts/astar_planner_py.py
  scripts/path_recorder.py
  scripts/path_follower.py
  scripts/multi_view.py
  scripts/coverage_path.py
  # scripts/occupancy_map.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  FILES_MATCHING PATTERN "*.py"
)

# add_executable(${PROJECT_NAME} src/main.cpp src/Astar.cpp src/OccMapTransform.cpp)
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})