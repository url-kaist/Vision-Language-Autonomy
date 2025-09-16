//
// Created by dshong on 25. 4. 23.
//

#include "snapshot.h"


Snapshot::Snapshot(ros::NodeHandle nh) : nh_(nh) {
    log("Snapshot created");
    try{
      for (const auto& entry : fs::directory_iterator(snapshot_dir_)) {
        fs::remove(entry.path());
      }
      log("All files in " + snapshot_dir_ + " have been deleted");
    } catch (std::exception& e) {
      ROS_ERROR("Failed to create snapshot: %s", e.what());
    }

    if (std::remove(json_write_path_.c_str()) == 0){
      log("remove " + json_write_path_ + " snapshot");
    } else {
      log("failed to remove " + json_write_path_ + " snapshot");
    }
    if (std::remove(json_read_path_.c_str()) == 0){
      log("remove " + json_read_path_ + " snapshot");
    } else {
      log("failed to remove " + json_read_path_ + " snapshot");
    }
}

Snapshot::~Snapshot() {}

void Snapshot::odomCallback(const nav_msgs::OdometryConstPtr& msg){
    const geometry_msgs::Point &curr = msg->pose.pose.position;

    if (!last_pose_initialized_){
        last_pose_ = curr;
        last_pose_initialized_ = true;
        return;
    }

    double dist = distance(curr, last_pose_);
    if (dist > threshold_){
        last_pose_ = curr;
        takeSnapshot();
    }
}

void Snapshot::takeSnapshot(){
  log("takeSnapshot");
  if (!snapshot_client_.waitForExistence(ros::Duration(1.0))){
    return;
  }

  snapshot::SnapshotData srv;
  if (snapshot_client_.call(srv)){
    saveImage(srv.response.image, false);
    saveImage(srv.response.image_annotated, true);
    saveObjects(srv.response.objects);
  }
}

void Snapshot::saveImage(const sensor_msgs::Image& msg, bool with_objects){ // const std::vector<snapshot::MapObject>& objects, bool with_objects){
  try{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    std::stringstream ss;
    if (not with_objects){
      ss << snapshot_dir_ << std::fixed << std::setprecision(3) << msg.header.stamp.toSec() << ".jpg";
      last_image_path_ = ss.str();
    }

    std::string filename = last_image_path_;
    if (with_objects){
      std::string target = ".jpg";
      std::string replacement = "_annotated.jpg";
      size_t pos = filename.rfind(target);
      if (pos != std::string::npos && pos + target.length() == filename.length()){
        filename.replace(pos, target.length(), replacement);
      }
    }

    cv::imwrite(filename, cv_ptr->image);
    log("Image saved: " + std::string(filename));
  } catch (cv_bridge::Exception& e){
    ROS_ERROR("%s", e.what());
  }
}

void Snapshot::saveObjects(const std::vector<snapshot::MapObject>& objects){
  log("saveObjects");
  Json::Value new_entry;
  new_entry["image"] = last_image_path_;
  for (const auto &obj : objects){
    Json::Value o;
    o["id"] = obj.id;
    o["class_id"] = obj.class_id;
    o["class_name"] = obj.class_name;
    o["center"] = pointToJson(obj.center);
    o["min_bbox"] = pointToJson(obj.min_bbox);
    o["max_bbox"] = pointToJson(obj.max_bbox);
    new_entry["objects"].append(o);
  }

  // Try to open and parse existing file
  Json::Value all_data;
  std::ifstream infile(json_write_path_, std::ifstream::binary);
  if (infile.good()) {
    infile >> all_data;  // Read existing content
    infile.close();

    if (!all_data.isArray()) {
      log("Warning: existing JSON file is not an array. Resetting.");
      all_data = Json::Value(Json::arrayValue);
    }
  } else {
    all_data = Json::Value(Json::arrayValue);
  }
  all_data.append(new_entry);

  // Save back to file
  std::ofstream outfile(json_write_path_);
  outfile << all_data.toStyledString();
  outfile.close();

  try{
    fs::copy_file(json_write_path_, json_read_path_, fs::copy_options::overwrite_existing);
  } catch (std::exception& e) {
    ROS_ERROR("%s", e.what());
  }

  log("Object metadata saved " + std::string(json_write_path_));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Snapshot");
    ros::NodeHandle nh;

    Snapshot snapshot(nh);
    ros::spin();

    return 0;
}
