#!/usr/bin/python3
import sys
sys.path.append('/ws/external/ai_module/devel/lib/python3/dist-packages')
sys.path.append('/ws/external/ai_module/src/task_planner/scripts')

import time
import argparse
import json

from ai_module.src.task_planner.scripts.utils import *
from ai_module.src.visual_grounding.scripts.vlms.loaders.client import LlmClient
from ai_module.src.visual_grounding.scripts.vlms import model_name_to_type, CONFIG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # realpath
PARENT_DIR = os.path.dirname(CURRENT_DIR)

class TaskPlanner():
    def __init__(self):
        self.task = None
        self.plan = None

        # Load config from config.json
        with open("/ws/external/ai_module/src/config.json", "r") as f:
            config = json.load(f)

        model_name = config['MODEL_NAME']
        vlm_type = model_name_to_type(model_name)
        api_key_name = CONFIG[vlm_type]['API_KEY_NAME']
        self.client = LlmClient(model_name=model_name, api_key=config[f'{api_key_name}0'])

    def set_task(self, task):
        self.task = task

    def create_subtasks(self):
        # get prompt
        prompt = get_prompt(self.task)

        # make messages
        messages = [
            {"role": "system", "content": "You are an expert Python code generator. You MUST output ONLY a complete Python function named 'create_subtasks_by_LLM()'. The function must start with 'def create_subtasks_by_LLM():' and all code must be properly indented inside this function. Do not include any markdown formatting, explanations, or text outside the function. The function must be executable without any syntax errors."},
            {"role": "user",
             "content": prompt}
        ]

        # query LM
        response, _, usage_log = self.client.get_response(messages, 0)

        # rm markdown
        response = response.replace("```python", "").replace("```", "")

        # run the generated code
        try:
            exec(response) # define the function
            exec("plan = create_subtasks_by_LLM()") # run the generated function
            exec("self.plan = plan")
        except Exception as e:
            print(f"Error executing LLM response: {e}")
            print(f"Response was: {response}")
            raise e

    @property
    def output_dict(self):
        def convert(obj):
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            else:
                return obj.to_dict()

        return {
            "question": self.task,
            "constraints": convert(self.plan.constraints),
            "steps": convert(self.plan.steps),
        }

    def save_subtasks(self, filename):
        save_json(self.output_dict, filename)

    def get_plan(self):
        return self.plan


def validate_plan(plan):
    for t in plan.constraints + plan.steps:
        if not isinstance(t, Task):
            raise TypeError(f"Invalid task type: {type(t)}")
        if not isinstance(t.entity, Entity):
            raise TypeError(f"Invalid entity type in task {t.action}: {type(t.entity)}")


def run_task_planner(question, save_dir=os.path.join(PARENT_DIR, "output"), save=True, file_name="output.json"):
    # create folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory {save_dir} created.")
    
    # create task planner
    task_planner = TaskPlanner()

    # measure time
    start_time = time.time()

    # set task and create subtasks
    task_planner.set_task(question)
    task_planner.create_subtasks() # using LLM

    # measure time
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f} sec")

    if save:
        task_planner.save_subtasks(os.path.join(save_dir, file_name))

    # get subtasks
    plan = task_planner.get_plan()
    validate_plan(plan)
    return plan


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--option",
                      type=str,
                      default="input_question",
                      choices=['test_question', 'question_from_file', 'input_question', ''],
                      help="1: run task planner, 2: run task planner from file")
    args = args.parse_args()
    
    save_dir = os.path.join(PARENT_DIR, "output")
    

    if args.option == "test_question":
        ## option1: run task planner from test question
        # question = "How many black trash cans are near the window?"
        # question = "Find the orange chair between the table and sink that is closest to the window."
        question = "First, go to the vase closest to the easel, then, take the path between the couch and the table and stop at the window closest to the couch."
        # question = "Take the path near the window to the fridge, avoid the path between the two tables and go near the blue trash can near the window."
        plan = run_task_planner(question, save_dir=save_dir)
        print(plan)
    else:
        print("hello")
