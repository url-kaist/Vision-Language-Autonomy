//
// Created by dshong on 25. 4. 8.
//
#include <ros/ros.h>
#include "user_interface.h"


int main(int argc, char** argv)
{
    printf("Hello UserInterface :)\n");
    ros::init(argc, argv, "UserInterface");
    ros::NodeHandle nh;

    UserInterface userInterface(nh);

    ros::spin();

    return 0;
}