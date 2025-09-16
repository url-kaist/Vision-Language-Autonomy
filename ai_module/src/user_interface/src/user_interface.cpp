//
// Created by dshong on 25. 4. 8.
//
#include "user_interface.h"


const std::string Q1 = "Find the bed under the butterfly picture.";
const std::string Q2 = "Count the number of pillows on the sofa.";
const std::string Q3 = "Pass between the dining table and the Irish table and stop in front of the kitchen hood.";
const std::string Q4 = "Find the plant closest to the chair.";
const std::string Q5 = "Count the number of wine bottles on the Irish table.";
const std::string Q6 = "Avoid the TV and the living room table, go to the jar next to the TV, then go to the trash can in the bathroom.";
const std::string Q7 = "Take the path between the sofa and the coffee table.";
const std::string Q8 = "Take the path near the purple pillow on the sofa.";
const std::string Q9 = "Avoid the path between the sofa and the coffee table and go to the kettle on the dining table, then go to the potted plant between the curtain and the TV.";
const std::string Q10 = "Avoid the path near the purple pillow on the sofa and go to the kettle on the dining table, then go to the potted plant between the curtain and the TV.";
const std::string Q11 = "Take the path between the sofa and the coffee table and go to the bed under the butterfly picture, then go to the speaker on the TV cabinet closest to the potted plant on the TV cabinet.";
const std::string Q12 = "Take the path between the sofa and the coffee table and go to the big potted plant next to the TV, then go to the speaker on the TV cabinet closest to the potted plant on the TV cabinet.";

void UserInterface::timerCallback(const ros::TimerEvent& event) {
    if (!ros::ok()){
        return;
    }
    if (is_running_ || subscribe_mode_){
        return;
    }

    is_running_ = true;

    if (debug_) {
        printf("Let's subscribe from /challenge_question!\n");
        subscribe_mode_ = true;
        return;
    }

    printf("\n\x1B[32mPlease choose the type of question:\n");
    printf("  0) Subscribe from /challenge_question\n");
    printf("     You can publish using `rostopic pub /challenge_question std_msgs/String \"Find a blue chair.\"`.");
    printf(" [Easy]\n");
    printf("  1) %s\n", Q1.c_str());
    printf("  2) %s\n", Q2.c_str());
    printf("  3) %s\n", Q3.c_str());
    printf(" [Hard]\n");
    printf("  4) %s\n", Q4.c_str());
    printf("  5) %s\n", Q5.c_str());
    printf("  6) %s\n", Q6.c_str());
    printf("  7) %s\n", Q7.c_str());
    printf("  8) %s\n", Q8.c_str());
    printf("  9) %s\n", Q9.c_str());
    printf(" 10) %s\n", Q10.c_str());
    printf(" 11) %s\n", Q11.c_str());
    printf(" 12) %s\n", Q12.c_str());
    printf("  Or, you can just type your own question!!\n");
    printf("\x1B[0m\n");

    std::string input;
    if (!std::getline(std::cin >> std::ws, input)) {
        is_running_ = false;
        return;
    }

    if (input == "0"){
        printf("Let's subscribe from /challenge_question!\n");
        subscribe_mode_ = true;
        return;
    } else if (input == "1") { input = Q1; }
    else if (input == "2") { input = Q2; }
    else if (input == "3") { input = Q3; }
    else if (input == "4") { input = Q4; }
    else if (input == "5") { input = Q5; }
    else if (input == "6") { input = Q6; }
    else if (input == "7") { input = Q7; }
    else if (input == "8") { input = Q8; }
    else if (input == "9") { input = Q9; }
    else if (input == "10") { input = Q10; }
    else if (input == "11") { input = Q11; }
    else if (input == "12") { input = Q12; }
    else { printf("Invalid choice.\n"); }

    user_interface::SetInput srv_set_input;
    srv_set_input.request.instructions = input;
    if (!input.empty() && user_interface_client_.call(srv_set_input) && srv_set_input.response.success) {
        printf(("Received message: " + srv_set_input.response.message).c_str());
    } else {
        ROS_ERROR("Failed to set input!");
    }

    is_running_ = false;
}

void UserInterface::challengeQuestionCallback(const std_msgs::String::ConstPtr& msg) {
    std::string input = msg->data;

    user_interface::SetInput srv_set_input;
    srv_set_input.request.instructions = input;
    srv_set_input.request.current_time = ros::Time::now();
    if (!input.empty() && user_interface_client_.call(srv_set_input) && srv_set_input.response.success) {
        printf(("Received message: " + srv_set_input.response.message).c_str());
    } else {
        ROS_ERROR("Failed to set input!");
    }
}
