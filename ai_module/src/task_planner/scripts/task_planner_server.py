#!/usr/bin/env python3
# python3 /ws/external/ai_module/src/task_planner/scripts/task_planner_server.py
import os
import sys
sys.path.append("/ws/external")
sys.path.append('/ws/external/ai_module/devel/lib/python3/dist-packages/task_planner')
import rospy
import subprocess

from srv import GetSubplans, GetSubplansResponse

sys.path.append('/ws/external/ai_module/src/task_planner/scripts')
from task_planner_main import run_task_planner

from ai_module.src.utils.logger import Logger


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # realpath
PARENT_DIR = os.path.dirname(CURRENT_DIR)


if __name__ == "__main__":
    rospy.init_node('task_planner_server')
    quiet = rospy.get_param('~quiet', False)
    logger = Logger(quiet=quiet, prefix="TaskPlanner", log_path="/ws/external/log/task_planner.log")

    def handle_get_subplans(req):
        logger.loginfo(f"Received question: {req.question}")

        plan = run_task_planner(req.question)
        rospy.loginfo(plan)

        command = None
        if plan.type in [0, 1]:
            logger.loginfo("Plan type is 0 or 1. Launching 'vg.launch'...")
            command = ['roslaunch', 'visual_grounding', 'vg.launch']
        elif plan.type == 2 and len(plan.constraints) > 0:
            logger.loginfo("Plan type is 2. Launching 'vg_if_two.launch'...")
            command = ['roslaunch', 'visual_grounding', 'vg_if_two.launch']
        elif plan.type == 2 and len(plan.constraints) == 0:
            logger.loginfo("Plan type is 2. Launching 'vg_if_one.launch'...")
            command = ['roslaunch', 'visual_grounding', 'vg_if_one.launch']

        logger.loginfo("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.loginfo(f"Command: {command}")
        logger.loginfo(f"plan type: {plan.type}")
        logger.loginfo(f"plan constraints: {plan.constraints}")
        logger.loginfo(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

        if command:
            vg_process = subprocess.Popen(command, preexec_fn=os.setsid)
            logger.loginfo(f"Started new process with PID: {vg_process.pid}")

        # Get response
        response = GetSubplansResponse()
        response.success = True
        response.message = "Subplans are generated successfully."
        response.plan = plan.to_msg()

        return response

    service = rospy.Service('/task_planner/request_subplans', GetSubplans, handle_get_subplans)
    logger.loginfo("Task Planner Service Ready.")
    rospy.spin()
