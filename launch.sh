#!/bin/bash
export OFFLINE_MAP_DIR=/ws/data/VLA/offline_map
export KEYFRAMES_DIR=/ws/data/VLA/keyframes
#export KEYFRAMES_DIR2=/ws/external/test_data/vla_js_chair_2025-12-17-12-17-43/keyframes
export TRAVERSABLE_PATH=/ws/data/VLA/E3_3225_TRIP.pcd
sudo chmod 777 -R /ws/external/
sudo chmod 777 -R /ws/data/
export PATH="$HOME/.local/bin:$PATH"

MODE="$1"
shift                     # 이제부터 남은 것들은 roslaunch args (ex: real_world:=true ...)

cd /ws/external/ai_module
if [ "$MODE" == "sg" ]; then
  catkin_make && cd /ws/external
  source /ws/external/ai_module/devel/setup.bash

  # cd /ws/external/system/unity/ && catkin_make &&
  # ./system_bring_up.sh &
  # sleep 5
  cd /ws/external/ai_module
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
