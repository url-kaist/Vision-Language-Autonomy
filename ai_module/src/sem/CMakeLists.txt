cmake_minimum_required(VERSION 3.16)
project(sem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED
  roscpp
  rospy
  cv_bridge
  image_transport
  message_generation
  std_msgs
  geometry_msgs
  snapshot
)

catkin_python_setup()

add_message_files(
  FILES
  ObjectNode.msg
  ObjectNodes.msg
)


add_service_files(
  FILES
  ObjectNodeFromClass.srv
  SetClasses.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport roscpp message_generation
  snapshot
  LIBRARIES ${PROJECT_NAME}
)
