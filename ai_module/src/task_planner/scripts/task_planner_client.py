import rospy

from task_planner.msg import Plan, Node, Edge, Task
from task_planner.srv import GetSubplans, GetSubplansResponse

def call_service():
    rospy.init_node('task_planner_client')
    rospy.wait_for_service('/task_planner/request_subplans')

    try:
        get_subplans = rospy.ServiceProxy('/task_planner/request_subplans', GetSubplans)
        # TODO: question should be inputted by users.
        question = "Count the number of pillows on the sofa."
        response = get_subplans(question)

        rospy.loginfo(response.plan)

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    call_service()
