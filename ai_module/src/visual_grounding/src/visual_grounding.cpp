//
// Created by dshong on 25. 4. 7.
//
#include "visual_grounding.h"


VisualGrounding::VisualGrounding(ros::NodeHandle nh) : nh_(nh) {
    requestTaskService_ = nh_.advertiseService(node_name_ + "/request_task", &VisualGrounding::requestTaskCallback, this);
}

VisualGrounding::~VisualGrounding() {}

bool VisualGrounding::requestTaskCallback(visual_grounding::RequestTask::Request &req, visual_grounding::RequestTask::Response &res){
    ROS_INFO("[VisualGrounding::requestTaskCallback] ^^*");
    target_entities_ = req.target_entities;
    tasks_ = req.subtask;

    res.success = true;
    res.message = "Subtask is stored successfully";
    return true;
}


