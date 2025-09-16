//
// Created by dshong on 25. 2. 17.
//

#ifndef TASK_PLANNER_H
#define TASK_PLANNER_H

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <string>
#include <vector>
#include <thread>

class TaskPlanner {
public:
    TaskPlanner();
    ~TaskPlanner();
    void questionCallback(const std_msgs::String::ConstPtr& question);

private:
    ros::NodeHandle nh_;
    const std::string node_name_ = "/task_planner";
    const std::string question_topic_ = "question";
    const std::string subplans_topic_ = "task_planner/subplans";

    // Service
    ros::ServiceServer startService_;
    bool startCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    // Subscribers and Clients
    ros::Subscriber question_sub_;
    ros::Publisher subplans_pub_;
    ros::ServiceClient subplans_client_;

};
#endif //TASK_PLANNER_H
