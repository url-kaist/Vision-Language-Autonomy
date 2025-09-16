//
// Created by dshong on 25. 4. 8.
//

#ifndef MAPPING_H
#define MAPPING_H

#include <fstream>
#include <mutex>
#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Pose2D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>  // Optional, for iterating easily
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <deque>
#include <map>
#include <nlohmann/json.hpp>

#include "user_interface/SetInput.h"
#include "task_planner/Subplans.h"
#include "task_planner/GetSubplans.h"
#include "visual_grounding/SetSubplans.h"
#include "sem/SetClasses.h"
#include "utils/logger.h"

using json = nlohmann::json;


enum TaskState {
    Idle, // Initial state, no input
    InstructionReady,
    InstructionReceived, // User inputs instructions
    SubplansReady, // Subplans are ready to be processed
    SubplansGenerated, // Subplans are generated based on instructions
    Processing, // Subplans are processing
    Done,
};


std::string taskStateToString(TaskState state) {
  switch (state) {
    case TaskState::Idle: return "Idle";
    case TaskState::InstructionReady: return "InstructionReady";
    case TaskState::InstructionReceived: return "InstructionReceived";
    case TaskState::SubplansReady: return "SubplansReady";
    case TaskState::SubplansGenerated: return "SubplansGenerated";
    case TaskState::Processing: return "Processing";
    case TaskState::Done: return "Done";
    default: return "UNKNOWN";
  }
}

enum class WorkKind { None, Constraint, Step };

struct AssignedTask {
  task_planner::Task task;
  WorkKind kind; // Constraint or Step
};

struct VGNode {
  std::string ns;  // e.g., "/visual_grounding_0"
  ros::ServiceClient set_subplans_cli;
  ros::ServiceClient status_cli;  // Service to check the node status
  ros::ServiceClient reset_cli;   // Service to reset the node
  ros::ServiceServer active_signal_srv; // Service to signal active VG node

  std::vector<std::string> api_keys; // API keys for the node
  bool connected = false;
  bool busy = false;
  bool is_active = false;  // Whether this node is active in the current task
  size_t step_idx = -1; // Current step index for this node // -1 means not assigned
  WorkKind last_kind = WorkKind::None;
};


class Manager {
public:
    Manager(ros::NodeHandle& nh);
    ~Manager();
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Timer timer_;
    ros::Time startTime;

    Logger logger_;

    // ROS node name
    const std::string node_name_mapping_ = "/scene_graph";
    const std::string node_name_grounding_ = "/visual_grounding";
    const std::string node_name_task_planning_ = "/task_planner";
    const std::string node_name_user_input_ = "/user_interface";

    // Publishers
    ros::Publisher timeout_pub_ = nh_.advertise<std_msgs::Empty>("/timeout", 1);
    ros::Publisher goal_pub_ = nh_.advertise<geometry_msgs::Pose2D>("/way_point_with_heading", 1);
    ros::Publisher steps_pub_ = nh_.advertise<std_msgs::String>("/steps", 1); // Subscriber: exploration
    ros::Publisher current_step_idx_pub_ = nh_.advertise<std_msgs::Int16>("/current_step_idx", 1); // Subscriber: exploration
    ros::Publisher question_type_pub_ = nh_.advertise<std_msgs::String>("/question_type", 1); // Subscriber: exploration, path_follower
    ros::Publisher active_nodes_pub_ = nh_.advertise<std_msgs::String>("/active_nodes", 1); // Publisher for active VG nodes
    
    // Service
    ros::ServiceServer srv_set_input_server_;
    bool setInput(user_interface::SetInput::Request &req, user_interface::SetInput::Response &res);
    ros::ServiceClient srv_get_subplans_client_;
    ros::ServiceClient mapping_client_add_classes_;

    // Functions
    bool startSystem();
    void timerCallback(const ros::TimerEvent& event);
    std::string getActiveNodesString(); // New function to get active nodes as string
    void publishActiveNodes(); // New function to publish active nodes

    // Flags
    bool isRunning_ = false;
    bool isStarted_;
    TaskState task_state_ = Idle;
    bool hasTask_ = false;
    bool goal_in_progress_ = false;
    bool read_detected_objects_ = false;

    // Variables
    std::string instructions_;
    ros::Time start_time_;
    task_planner::Plan plans_;
    task_planner::Task current_step_;
    geometry_msgs::Pose2D target_pose_;

    // === Multi-VG scheduler ===
    std::vector<VGNode> vg_nodes_;
    size_t max_vg_nodes_ = 0;           // max number of VG nodes
    size_t connected_vg_nodes_ = 0;     // number of connected VG nodes
    size_t active_step_node_idx_ = -1;  // index of the active step node
    size_t current_step_idx_ = -1;      // index of the current step in the plan
    size_t assigned_step_idx_ = -1;     // index of the assigned step in the plan
    size_t max_step_idx_ = -1;          // maximum step index in the plan

    std::deque<AssignedTask> task_queue_;

    std::vector<std::string> all_api_keys_;
    int num_clients_per_node_ = 1;

    bool loadConfigAndComputeMaxNodes(const std::string& path);
    size_t discoverVGNodes();
    void splitApiKeysPerNode();
    bool sendSubplanToNode(VGNode& node, const task_planner::Task& current_step, WorkKind kind);
    void assignInitialSubplans(); 
    void pollAndReassign();
    void chooseActiveFromBusyStepNodes();
    void setNextActiveStepNode();
    bool callStatus(VGNode& node, bool& done);
    bool callReset(VGNode& node);  
    bool resetManager();
    void idleVGNode(VGNode& node);
};

#endif //MAPPING_H
