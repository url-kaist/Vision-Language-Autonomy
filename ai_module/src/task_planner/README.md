## Setting

1. `ai_module/src/task_planner/scripts/.env`를 생성해서 아래 코드를 붙여넣기
    
    ```bash
    OPENAI_API_KEY=<your_openai_api_key>
    ```
    
    - ⭐ [참고] OPENAI_API_KEY 사용 방법
        - 보안 및 관리를 위해서 `ai_module/src/vlm/scrips/`아래에 credentials 패키지를 만듦.
        - `scrips/` 아래에 있는 python project는 아래와 같은 코드로 OEPNAI_API_KEY를 불러와 사용할 수 있음.
            
            ```python
            from credentials import OPENAI_API_KEY
            ```
            
2. bashrc 수정
    
    ```bash
    echo 'export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src/task_planner/scripts' >> ~/.bashrc
    source ~/.bashrc
    ```
    

## How to Run?

### (Main) 

### Example1. Simple Python-Python Service
1. At terminal 1
    ```bash
    roscore
    ```
2. At terminal 2 (Python Server)
    ```bash
    cd /ws/external/ai_module
    catkin_make
    source devel/setup.bash
    rosrun task_planner task_planner_server.py
    ```
1. At terminal 3 (Python Client)
    ```bash
    cd /ws/external/ai_module
    catkin_make
    source devel/setup.bash
    rosrun task_planner task_planner_client.py
    ```

### Example2. Simple Python-C++ Service
1. At terminal 1
    ```bash
    roscore
    ```
2. At terminal 2 (Python Server)
    ```bash
    cd /ws/external/ai_module
    catkin_make
    source devel/setup.bash
    rosrun task_planner task_planner_server.py
    ```
1. At terminal 3 (Python Client)
    ```bash
    cd /ws/external/ai_module
    catkin_make
    source devel/setup.bash
    rosrun task_planner task_planner_client
    ```

### Python

- 설정된 예제 question으로 subtasks 생성
    
    ```bash
    cd /ws/external/ai_module/src/ 
    python task_planner/scripts/task_planner_main.py --option "test_question"
    ```
    
    - 위 코드를 돌리면 task_planner.py에 입력된 question을 바탕으로 `task_planner/output/` 경로에 output.json 파일이 생성됨.
- Generate subtasks for your question
    
    ```bash
    cd /ws/external/ai_module/src/ 
    python task_planner/scripts/task_planner_main.py --option "input_question"
    ```
    
    - 위 코드를 돌리면 task_planner.py에 입력된 question을 바탕으로 `task_planner/output/` 경로에 output.json 파일이 생성됨.
- 대회에서 제공해준 questions.json 파일로 subtasks 생성
    
    ```bash
    cd /ws/external/ai_module/src/ 
    python task_planner/scripts/task_planner_main.py --option "question_from_file"
    ```
    
    - 위 코드를 돌리면 `task_planner/output/questions.json`을 읽어 {scene}_{qtype}_{idx}.json 파일들이 생성됨.
    - ⭐ [참고] task planning하는데 질문 1개당 평균적으로 걸리는 시간 (25.03.01.)
        
        ```bash
        Average processing time: 8.40 sec
        ```
        

## Visualization

```bash
cd /ws/external/ai_module/src/
python task_planner/scripts/visualize_output.py
```

- 위 코드를 실행하면 `output/`에 읽는 모든 json 파일들을 시각화해서 `visualization/`에 저장함.
    - example
        
        ![chinese_room_instruction_following_1.png](../../../figures/chinese_room_instruction_following_1.png)
