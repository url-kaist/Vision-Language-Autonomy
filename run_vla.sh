#!/bin/bash
set -e

SENSOR=${1:-RGBD} # TODO: RGBD | LIDAR
MODE=${2:-sim} # TODO: sim | real_world
GRID_SIZE=${3:-20}
DELAY=${4:-10}

# MODE: sim | real_world  →  ROS param real_world:=true/false
if [ "$MODE" = "real_world" ]; then
  REAL_WORLD=true
else
  REAL_WORLD=false
fi
LAUNCH_ARG="real_world:=${REAL_WORLD}"


SESSION="vla_session"
WORKDIR="/ws/external"

TITLE_RED="#[fg=red,bold]"
TITLE_BLUE="#[fg=blue,bold]"
TITLE_MAGENTA="#[fg=magenta,bold]"
TITLE_YELLOW="#[fg=yellow,bold]"
RESET="#[default]"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# 세션/윈도우 생성 (시작 디렉토리 고정)
tmux new-session -d -s "$SESSION" -n "VLA_Project" -c "$WORKDIR"

# pane 종료돼도 창 유지(로그 확인용)
tmux set -g remain-on-exit on

# 타이틀 바 표시 + "pane_title" 대신 "@label" 사용
tmux setw -t "$SESSION:0" pane-border-status top
tmux setw -t "$SESSION:0" pane-border-format " #{@label} "

# ---- pane 만들기 (pane_id를 잡아두면 번호 꼬임 없음) ----
P0=$(tmux display-message -p -t "$SESSION:0.0" "#{pane_id}")

P1=$(tmux split-window -h -t "$P0" -c "$WORKDIR" -P -F "#{pane_id}")
P2=$(tmux split-window -v -t "$P0" -c "$WORKDIR" -P -F "#{pane_id}")
P3=$(tmux split-window -v -t "$P1" -c "$WORKDIR" -P -F "#{pane_id}")

# ---- 각 pane 라벨(@label) 설정 (이게 border에 뜸) ----
tmux set-option -pt "$P0" @label "${TITLE_RED}1. Scene Graph (${SENSOR}/${MODE})${RESET}"
tmux set-option -pt "$P1" @label "${TITLE_BLUE}2. NAV${RESET}"
tmux set-option -pt "$P2" @label "${TITLE_MAGENTA}3. MAIN (after ${DELAY}s)${RESET}"
tmux set-option -pt "$P3" @label "${TITLE_YELLOW}4. QUESTION PUBLISHER${RESET}"

# ---- 커맨드 실행 ----
# tmux send-keys -t "$P0" "./launch.sh sg" C-m
# tmux send-keys -t "$P1" "./launch.sh nav" C-m
# tmux send-keys -t "$P2" "sleep ${DELAY} && ./launch.sh main" C-m

# tmux send-keys -t "$P0" "./launch.sh sg $LAUNCH_ARG" C-m
# tmux send-keys -t "$P1" "./launch.sh nav $LAUNCH_ARG" C-m
# tmux send-keys -t "$P2" "sleep ${DELAY} && ./launch.sh main $LAUNCH_ARG" C-m

tmux send-keys -t "$P0" "bash -lc 'cd $WORKDIR && ./launch.sh sg $LAUNCH_ARG'" C-m
tmux send-keys -t "$P1" "bash -lc 'cd $WORKDIR && ./launch.sh nav $LAUNCH_ARG'" C-m
tmux send-keys -t "$P2" "bash -lc 'cd $WORKDIR && sleep ${DELAY} && ./launch.sh main $LAUNCH_ARG'" C-m



# 질문 publish는 "입력만" 해두고, 사용자가 Enter 치게
# tmux send-keys -t "$P3" "rostopic pub -1 /challenge_question std_msgs/String \"data: 'Find the pillow closest to the book on the stool.'\""

tmux send-keys -t "$P3" \
  "rostopic pub -1 /challenge_question std_msgs/String \"data: \\\"Find the chair with a blue seat.\\\"\""


# 레이아웃 정리
tmux select-layout -t "$SESSION:0" tiled

# 접속
tmux attach-session -t "$SESSION"