#!/usr/bin/env bash
set -euo pipefail

TOPIC="/challenge_question"
MSG_TYPE="std_msgs/String"

SCENARIOS=(
  "Find a red chair between blue chairs"
  "Find a red chair below the halloween poster"
  "Find a blue fire extinguisher next to the TV monitor"
  "How many books are on the bookshelf?"
  "How many chairs does the doll sit on?"
  "How many silver fire extinguishers are there next to the TV monitor?"
  "How many chairs below the halloween poster"
  "Find a chair below the christmas poster"
  "Find a chair below the tree poster"
  "Find a silver extinguisher next to the blue extinguisher" # 10
  "Find the chair below the halloween poster" # 11
  "How many fans on the desk" # 12
  "How many potted plants on the desk" # 13
  "Find the chair below the orange poster" # 14
  "How many fans on the drawer" # 15
  "Find a potted plant with orange flower" # 16
  "Find a chair below the halloween poster" # 17
  "How many white fans on the drawer" # 15 ** 
  "How many empty chairs?" # 19
  # 여기에 시나리오를 계속 추가하세요.
)

print_menu() {
  echo "Select a scenario to publish to ${TOPIC}:"
  for i in "${!SCENARIOS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${SCENARIOS[$i]}"
  done
  echo "  [q] Quit"
}

publish_scenario() {
  local text="$1"
  echo "Publishing: ${text}"
  # ROS1 rostopic pub 형식 (질문에 주신 그대로)
  rostopic pub -1 "${TOPIC}" "${MSG_TYPE}" "data: '${text}'"
}

while true; do
  print_menu
  read -r -p "Enter choice: " choice

  if [[ "${choice}" == "q" || "${choice}" == "Q" ]]; then
    echo "Bye."
    exit 0
  fi

  # 숫자 입력 검증
  if [[ "${choice}" =~ ^[0-9]+$ ]]; then
    idx=$((choice - 1))
    if (( idx >= 0 && idx < ${#SCENARIOS[@]} )); then
      publish_scenario "${SCENARIOS[$idx]}"
      exit 0
    fi
  fi

  echo "Invalid choice. Try again."
  echo
done


