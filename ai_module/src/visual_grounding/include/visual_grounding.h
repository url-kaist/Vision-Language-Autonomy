//
// Created by dshong on 25. 4. 7.
//

#ifndef VISUAL_GROUNDING_H
#define VISUAL_GROUNDING_H

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <string>
#include <vector>
#include <thread>

#include "visual_grounding/RequestTask.h"
#include "task_planner/Entity.h"


class VisualGrounding {
public:
    VisualGrounding(ros::NodeHandle nh);
    ~VisualGrounding();

private:
    ros::NodeHandle nh_;

    // Names
    const std::string node_name_ = "/visual_grounding";

    // Service
    ros::ServiceServer requestTaskService_;

    // Subscribers

    // Functions
    bool requestTaskCallback(visual_grounding::RequestTask::Request &req, visual_grounding::RequestTask::Response &res);

    // Variables
    std::vector<task_planner::Entity> target_entities_;
    task_planner::SubTask tasks_;

};
#endif //VISUAL_GROUNDING_H
