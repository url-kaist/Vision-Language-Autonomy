#!/usr/bin/env python3
# python3 /ws/external/ai_module/src/task_planner/scripts/task_planner_server.py
import os
import sys
sys.path.append("/ws/external")
# sys.path.append('/ws/external/ai_module/devel/lib/python3/dist-packages/task_planner')
import rospy

from sem.srv import SetClasses, SetClassesResponse
from std_msgs.msg import Bool, String
from visual_grounding.srv import SetSubplans, SetSubplansResponse

# sys.path.append('/ws/external/ai_module/src/task_planner/scripts')
from utils import Edge, Node, Entity, Task, Plan

from ai_module.src.utils.logger import Logger


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # realpath
PARENT_DIR = os.path.dirname(CURRENT_DIR)


if __name__ == "__main__":
    rospy.init_node('test_task_planner_server')

    logger = Logger(quiet=False, prefix="TestTaskPlanner", log_path="/ws/external/log/test_task_planner.log")

    mapping_client_add_classes = rospy.ServiceProxy("/scene_graph/set_classes", SetClasses)
    steps_pub_ = rospy.Publisher("/steps", String, queue_size=1) # Subscriber: exploration
    set_subplans_cli = rospy.ServiceProxy("visual_grounding/set_subplans", SetSubplans)

    task_type = sys.argv[1]
    logger.loginfo(f"task_type: {task_type}")

    if task_type == 'find':
        instruction = "Find the  bed under the butterfly picture."

        plan = Plan()
        entity = Entity(target_name='the bed under the butterfly picture')
        entity.add_node(Node(1, name='bed', is_target=True))
        entity.add_node(Node(2, name='picture', is_target=False, attr={'subject': ['butterfly']}))
        entity.add_edge(Edge(name='under', source_id=1, target_ids=[2]))
        task = Task(action='find', entity=entity)
        plan.steps.append(task)
        logger.loginfo(f"plan:")
        logger.loginfo(f"{plan}")

        steps_msg = String()
        steps_msg.data = ""
        for step in plan.steps:
            steps_msg.data += f"/{step.entity.target_name}"
        steps_pub_.publish(steps_msg)
        logger.loginfo(f"Publish steps: {steps_msg}")

        # srv = SetSubplans()
        # srv.text_instruction = instruction
        # srv.constraints = plan.constraints
        # srv.steps = plan.steps
        # srv.current_step = plan.steps[0]
        plan_msg = plan.to_msg()
        logger.loginfo(f"plan_msg: {plan_msg}")

        set_subplans_cli(
            text_instruction=instruction,
            constraints=plan_msg.constraints,
            steps=plan_msg.steps,
            current_step=plan_msg.steps[0]
        )
    else:
        raise NotImplementedError('Not yet.')

    rospy.spin()