//
// Created by dshong on 25. 4. 8.
//
#include "sub_interface.h"


void SubInterface::challengeQuestionCallback(const std_msgs::String::ConstPtr& msg) {
    std::string input = msg->data;

    user_interface::SetInput srv_set_input;
    srv_set_input.request.instructions = input;
    if (!input.empty() && user_interface_client_.call(srv_set_input) && srv_set_input.response.success) {
        printf(("Received message: " + srv_set_input.response.message).c_str());
    } else {
        ROS_ERROR("Failed to set input!");
    }
}


int main(int argc, char** argv)
{
    printf("Hello UserInterface :)\n");
    ros::init(argc, argv, "UserInterface");
    ros::NodeHandle nh;

    SubInterface userInterface(nh);

    ros::spin();

    return 0;
}