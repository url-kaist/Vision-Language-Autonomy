#!/bin/bash
sudo chmod 777 -R /ws/external/
<<<<<<< HEAD

MODE="$1"
shift                     # 이제부터 남은 것들은 roslaunch args (ex: real_world:=true ...)

cd /ws/external/ai_module
if [ "$MODE" == "sg" ]; then
  catkin_make && cd /ws/external
  source /ws/external/ai_module/devel/setup.bash

=======
cd /ws/external/ai_module && catkin_make && cd /ws/external
source ai_module/devel/setup.bash

ARG="$2"
echo "$ARG"

cd ai_module
if [ "$1" == "sg" ]; then
>>>>>>> dev/cleanup_vg_dshong
  cd /ws/external/system/unity/ && catkin_make &&
  ./system_bring_up.sh &
  sleep 5
  cd /ws/external/ai_module
<<<<<<< HEAD
  roslaunch sem sg.launch "$@"
fi

if [ ! -f /ws/external/ai_module/devel/setup.bash ]; then
  echo "[launch.sh] ai_module not built yet. Run './launch.sh sg' first."
  exit 1
fi

if [ "$MODE" == "main" ]; then
  source /ws/external/ai_module/devel/setup.bash
  roslaunch vlm main.launch "$@"

elif [ "$MODE" == "nav" ]; then
  source /ws/external/ai_module/devel/setup.bash
  roslaunch exploration exploration.launch "$@"

elif [ "$MODE" == "benchmark" ]; then
  source /ws/external/ai_module/devel/setup.bash
  roslaunch vlm benchmark.launch "$@"

else
  echo "Usage: $0 [sg|main|nav|benchmark] [roslaunch_args...]"
  exit 1
fi
=======
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
>>>>>>> dev/cleanup_vg_dshong
