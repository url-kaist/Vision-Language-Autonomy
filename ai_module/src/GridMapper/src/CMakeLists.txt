cmake_minimum_required(VERSION 3.0.2)
project(nav_package)

## 컴파일에 필요한 ROS 패키지들을 찾습니다
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf
)

## 시스템 의존성 (Python Numpy 등은 보통 rospy가 알아서 처리하지만 명시 가능)
## catkin_python_setup() # setup.py가 있을 경우 주석 해제

################################################
## Declare ROS messages, services and actions ##
################################################

## (메시지나 서비스를 직접 정의하지 않았으므로 생략)

################################################
## catkin specific configuration              ##
################################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES trg_planner
#  CATKIN_DEPENDS rospy sensor_msgs nav_msgs geometry_msgs visualization_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## include 폴더가 있다면 추가
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## [중요] 파이썬 스크립트를 실행 가능한 프로그램으로 설치합니다.
## mapping.py가 현재 'maps' 폴더에 있다면 아래 경로를 "maps/mapping.py"로 바꾸세요.
## 하지만 가급적 'scripts' 폴더를 만들어서 그 안에 넣는 것을 추천합니다.
catkin_install_python(PROGRAMS
  scripts/mapping.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)