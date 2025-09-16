//
// Created by dshong on 25. 4. 8.
//

#ifndef USER_INPUT_H
#define USER_INPUT_H

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>

#include "user_interface/SetInput.h"


class SubInterface{

public:
    SubInterface(ros::NodeHandle& nh) : nh_(nh) {
        user_interface_client_ = nh_.serviceClient<user_interface::SetInput>("/user_interface/set_input");

        printf("Wait for existence...\n");
        user_interface_client_.waitForExistence();
        printf("UserInterface is generated.\n");
    };
    ~SubInterface(){};

private:
    ros::NodeHandle& nh_;
    ros::Timer timer_;

    // Subscribers
    bool subscribe_mode_ = false;
    void challengeQuestionCallback(const std_msgs::String::ConstPtr& msg);
    ros::Subscriber challenge_sub_ = nh_.subscribe("/challenge_question", 1, &SubInterface::challengeQuestionCallback, this);

    // Service
    ros::ServiceClient user_interface_client_;

    // Functions
    bool is_running_ = false;
};

#endif //USER_INPUT_H
