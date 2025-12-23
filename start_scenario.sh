#!/usr/bin/env bash
set -euo pipefail

TOPIC="/challenge_question"
MSG_TYPE="std_msgs/String"

SCENARIOS=(
  "Find a blue chair between red chairs"
  "Find a red chair below the halloween poster"
  "Find a blue fire extinguisher next to the TV monitor"
  "How many books are on the bookshelf?"
  "How many blur chairs are below the halloween poster?"
  "How many silver fire extinguishers are there next to the TV monitor?"
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


