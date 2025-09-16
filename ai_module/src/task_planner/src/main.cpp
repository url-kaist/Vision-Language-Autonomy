#include <iostream>
#include <cstdlib>
#include <ros/ros.h>
#include <std_msgs/String.h> // Message type
#include "task_planner.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "TaskPlanner");
    ros::NodeHandle nh;

    TaskPlanner taskPlanner;
    ros::spin();

    return 0;
}
