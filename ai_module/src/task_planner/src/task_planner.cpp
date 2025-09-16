//
// Created by dshong on 25. 2. 17.
//
#include "task_planner/GetSubplans.h"
#include "task_planner.h"
#include <task_planner/Subplans.h>

TaskPlanner::TaskPlanner() : question_topic_("/question"){
  startService_ = nh_.advertiseService(node_name_ + "/start", &TaskPlanner::startCallback, this);
}

TaskPlanner::~TaskPlanner() {}

bool TaskPlanner::startCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
  ROS_INFO("[TaskPlanner::startCallback] Setup subscribers...");
  question_sub_ = nh_.subscribe<std_msgs::String>(question_topic_, 10, &TaskPlanner::questionCallback, this);
  subplans_pub_ = nh_.advertise<task_planner::Subplans>(subplans_topic_, 10);

  ROS_INFO("[TaskPlanner::startCallback] Setup clients...");
  subplans_client_ = nh_.serviceClient<task_planner::GetSubplans>("get_subplans");

  ROS_INFO("[TaskPlanner::startCallback] Started successfully");
  res.success = true;
  res.message = node_name_ + " complete";
  return true;
}


void TaskPlanner::questionCallback(const std_msgs::String::ConstPtr& msg)
{
  std::string question = msg->data;
  ROS_INFO("Question: %s", question.c_str());

  // Create a service request
  task_planner::GetSubplans srv;
  srv.request.question = question;

  if (subplans_client_.call(srv)) {
    ROS_INFO("[Constraints]");
    for (const auto &constraint : srv.response.plan.constraints) {
      ROS_INFO("--- action %s ---", constraint.action.c_str());
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

  // Create message to publish
  task_planner::Subplans subplans_msg;
  subplans_msg.plan = srv.response.plan;

  // Publish task plan
  subplans_pub_.publish(subplans_msg);
  ROS_INFO("Publish subplans to: [%s]", subplans_pub_.getTopic().c_str());
}


