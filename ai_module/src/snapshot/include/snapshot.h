//
// Created by dshong on 25. 4. 23.
//

#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <Eigen/Dense>

#include <string>
#include <filesystem>
namespace fs = std::filesystem;

#include "snapshot/SnapshotData.h"
#include "snapshot/MapObject.h"

Json::Value pointToJson(const geometry_msgs::Point& pt) {
  Json::Value j;
  j["x"] = pt.x;
  j["y"] = pt.y;
  j["z"] = pt.z;
  return j;
}

class Snapshot {
public:
  Snapshot(ros::NodeHandle nh);
  ~Snapshot();

private:
  ros::NodeHandle nh_;

  // Names
  const std::string node_name_ = "/snapshot";
  const std::string snapshot_dir_ = "/ws/external/vis/snapshots/";
  const std::string json_write_path_ = "/ws/external/vis/snapshots/snapshots.json";
  const std::string json_read_path_ = "/ws/external/vis/snapshots.json";

  // Variables
  bool last_pose_initialized_ = false;
  geometry_msgs::Point last_pose_;
  double threshold_ = 3.0;
  std::string last_image_path_;

  // Subscribers
  ros::Subscriber odom_sub_ = nh_.subscribe("/state_estimation", 5, &Snapshot::odomCallback, this);
  ros::ServiceClient snapshot_client_ = nh_.serviceClient<snapshot::SnapshotData>("/SnapshotData");

  // Functions
  double distance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2){
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
  };
  void log(const std::string& text, const std::string& color = "\033[95m") {
    std::string func = __FUNCTION__; // This gets the name of the current function
    std::ostringstream msg;
    msg << color << "[Snapshot::" << func << "] " << text << "\033[0m";
    ROS_INFO_STREAM(msg.str());
  };
  void odomCallback(const nav_msgs::OdometryConstPtr& msg);
  void takeSnapshot();
  void saveImage(const sensor_msgs::Image& msg, bool with_objects); // , const std::vector<snapshot::MapObject>& objects, bool with_objects);
  void saveObjects(const std::vector<snapshot::MapObject>& objects);

};


#endif //SNAPSHOT_H
