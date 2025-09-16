#include <ros/ros.h>
#include <task_planner/GetSubplans.h>
#include <task_planner/Entity.h>
#include <task_planner/EntityRelation.h>
#include <task_planner/SubTask.h>

void callTaskPlannerService(int argc, char **argv) {
    ros::init(argc, argv, "task_planner_client");
    ros::NodeHandle nh;
    ros::service::waitForService("get_subplans");

    ros::ServiceClient client = nh.serviceClient<task_planner::GetSubplans>("get_subplans");
    task_planner::GetSubplans srv;
    // TODO: question should be inputted by users.
    srv.request.question = "TODO: question should be inputted by users.";

    if (client.call(srv)) {
        ROS_INFO("Question: %s", srv.request.question.c_str());
        ROS_INFO("[Constraints]");
        for (const auto &constraint : srv.response.plan.constraints) {
            ROS_INFO("--- action : %s ---", constraint.action.c_str());
            ROS_INFO("  entity : %s", constraint.entity.target_name.c_str());
        }
        ROS_INFO("[Steps]");
        for (const auto &step : srv.response.plan.steps) {
            ROS_INFO("--- action : %s ---", step.action.c_str());
            ROS_INFO("  entity : %s", step.entity.target_name.c_str());
        }
    } else {
        ROS_ERROR("Service call failed");
    }
}

int main(int argc, char **argv) {
    callTaskPlannerService(argc, argv);
    return 0;
}
