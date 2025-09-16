#include <iostream>
#include <list>
#include <cstdlib>
#include <ros/ros.h>
#include <std_msgs/String.h> // Message type
#include "visual_grounding.h"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "VisualGrounding");
    ros::NodeHandle nh;

    VisualGrounding visualGrounding(nh);

    ros::spin();

    return 0;
}
