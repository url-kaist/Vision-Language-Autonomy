#!/bin/bash
sudo chmod 777 -R /ws/external/
cd /ws/external/ai_module && catkin_make && cd /ws/external
source ai_module/devel/setup.bash

ARG="$2"
echo "$ARG"

cd ai_module
if [ "$1" == "sg" ]; then
  cd /ws/external/system/unity/ && catkin_make &&
  ./system_bring_up.sh &
  sleep 5
  cd /ws/external/ai_module
  roslaunch sem sg.launch $ARG
elif [ "$1" == "main" ]; then
  roslaunch vlm main.launch $ARG
elif [ "$1" == "nav" ]; then
  roslaunch exploration exploration.launch
elif [ "$1" == "benchmark" ]; then
  roslaunch vlm benchmark.launch $ARG
else
    echo "Usage: $0 [sg|main|nav]"
    exit 1
fi