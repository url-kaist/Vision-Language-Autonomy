//
// Created by dshong on 25. 4. 8.
//
#include "manager.h"

Manager::Manager(ros::NodeHandle& nh) : nh_(nh), pnh_("~") {
    bool quiet_;
    pnh_.param<bool>("quiet", quiet_, false);
    logger_.configure(quiet_, "Manager", "/ws/external/log/manager.log");

    startTime = ros::Time(0);
    // Load configuration and compute max VG nodes
    loadConfigAndComputeMaxNodes("/ws/external/ai_module/src/config.json");

    // Discover VG nodes
    // connected_vg_nodes_ = discoverVGNodes();

    // Split API keys per node
    // splitApiKeysPerNode();

    isStarted_ = startSystem();
}

Manager::~Manager() {
}

bool Manager::startSystem() {
    logger_.log("Start system...");


    srv_get_subplans_client_ = nh_.serviceClient<task_planner::GetSubplans>("/task_planner/request_subplans");

    



    srv_set_input_server_ = nh_.advertiseService("/user_interface/set_input", &Manager::setInput, this);

    

    // Wait for existence
    logger_.log("Wait for existence (task_planner, mapping)...");
    srv_get_subplans_client_.waitForExistence();
    



    logger_.log("Timer is created...");
    timer_ = nh_.createTimer(ros::Duration(1.0), &Manager::timerCallback, this);
    return true;
}


bool Manager::setInput(user_interface::SetInput::Request &req, user_interface::SetInput::Response &res){

    startTime = ros::Time::now();

    logger_.log(std::string("Received instruction: ") + req.instructions);
    instructions_ = req.instructions;
    start_time_ = req.current_time;
    task_state_ = InstructionReady;

    res.success = true;
    res.message = std::string("Success: ") + instructions_;
    return true;
}


void Manager::timerCallback(const ros::TimerEvent& event) {
    if (!ros::ok() && !isRunning_){
      return;
    }
    if (startTime.toSec() != 0.0){
        ros::Time currTime = ros::Time::now();
        std::cout<<"time collapse: "<<(currTime - startTime).toSec()<<std::endl;
        logger_.log("time collapse: " + std::to_string((currTime - startTime).toSec()));
        if ((currTime - startTime).toSec() > 600.0 - 30.0) {
            // Wait for 5 seconds after starting
            timeout_pub_.publish(std_msgs::Empty());
            logger_.log("Timeout: 9.5 minutes have passed since start.");
            resetManager();
            return;
        }
    }
    if (goal_in_progress_ == true){
        logger_.log("Move to target point...");
        goal_pub_.publish(target_pose_);
        return;
    }

    if (task_state_ == Idle){
        logger_.log(std::string("Task state:") + taskStateToString(task_state_));
    }

    if (task_state_ == InstructionReady){
        logger_.log(std::string("Task state: ") + taskStateToString(task_state_));

        mapping_client_add_classes_ = nh_.serviceClient<sem::SetClasses>(node_name_mapping_ + "/set_classes");
        mapping_client_add_classes_.waitForExistence();

        task_state_ = InstructionReceived;
    }

    if (task_state_ == InstructionReceived){
        logger_.log(std::string("Task state:") + taskStateToString(task_state_));
        isRunning_ = true;

        task_planner::GetSubplans srv_get_subplans;
        srv_get_subplans.request.question = instructions_;
        if (srv_get_subplans_client_.call(srv_get_subplans) && srv_get_subplans.response.success) {
            logger_.log("------------------------------");
            logger_.log("Task Planner: Convert instruction to subplans");
            std::set<std::string> class_set;
            logger_.log("[Constraints]");
            for (const auto &constraint : srv_get_subplans.response.plan.constraints) {
                logger_.log(std::string("--- action ") + constraint.action + std::string(" ---"));
                logger_.log(std::string("  entity : ") + constraint.entity.target_name.c_str());
                for (const auto &node : constraint.entity.relation_graph.nodes){
                    class_set.insert(node.name);
                }
            }

            logger_.log("[Steps]");
            for (const auto &step : srv_get_subplans.response.plan.steps) {
                logger_.log(std::string("--- action ") + step.action + std::string(" ---"));
                logger_.log(std::string("  entity : ") + step.entity.target_name.c_str());
                for (const auto &node : step.entity.relation_graph.nodes){
                    class_set.insert(node.name);
                }
            }
            std::vector<std::string> class_list(class_set.begin(), class_set.end());

            sem::SetClasses srv_set_classes;
            srv_set_classes.request.classnames = class_list;
            srv_set_classes.request.template_str = "{}";
            if (mapping_client_add_classes_.call(srv_set_classes) && srv_set_classes.response.success) {
                logger_.log("[Set Classnames]");
                for (const auto& cls: class_list){
                    logger_.log(" - " + cls);
                }
            } else {
                logger_.log("[Failed to Add Classnames] " + srv_set_classes.response.message);
            }

            logger_.log("------------------------------");
            plans_ = srv_get_subplans.response.plan;

            // TODO: Set the number of maximum trials
            if (plans_.steps.empty()){
                logger_.logerr("There isn't enough subtasks to execute. Regenerate plans...");
                return;
            }
            task_state_ = SubplansReady;
        } else {
            logger_.logerr("Failed to get subplans. Regenerate plans...");
            return;
        }
        isRunning_ = false;
    }

    if (task_state_ == SubplansReady){
        logger_.log(std::string("Task state:") + taskStateToString(task_state_));
        connected_vg_nodes_ = discoverVGNodes();
        splitApiKeysPerNode();

        task_state_ = SubplansGenerated;
    }

    if (task_state_ == SubplansGenerated){
        logger_.log(std::string("Task state:") + taskStateToString(task_state_));
        isRunning_ = true;

        // Clear the task queue and fill it with constraints and steps
        task_queue_.clear();
        for (const auto& c : plans_.constraints) task_queue_.push_back({c, WorkKind::Constraint});
        for (const auto& s : plans_.steps)       task_queue_.push_back({s, WorkKind::Step});
        max_step_idx_ = plans_.steps.size() - 1; // Set the maximum step index

        if (task_queue_.empty()){
            logger_.logerr("There is no task to execute. Regenerate plans...");
            isRunning_ = false;
            return;
        }

        // Assign the first subplan to the current step
        for (auto& node : vg_nodes_){
            if (!node.connected) continue;
            if (task_queue_.empty()) break;

            auto at = task_queue_.front(); task_queue_.pop_front();
            if (sendSubplanToNode(node, at.task, at.kind)){
                node.busy = true;
                node.last_kind = at.kind;
                node.is_active = (at.kind == WorkKind::Constraint); // Constraint nodes are always active
                node.step_idx = at.kind == WorkKind::Step ? ++assigned_step_idx_ : -1; // Set step index only for steps
            } else {
                logger_.logerr("Failed to send subplan to " + node.ns);
            }
        }

        if (assigned_step_idx_ == -1) {
            logger_.logerr("No VG node accepted subplans.");
            isRunning_ = false;
            return;
        }
        
        // Set the active VG nodes
        chooseActiveFromBusyStepNodes();

        task_state_ = Processing;
        hasTask_ = true;
    
        isRunning_ = false;
    }

    if (task_state_ == Processing){
        // Publish steps for exploration
        std_msgs::String steps_msg;
        steps_msg.data = "";
        for (const auto& step : plans_.steps){
            if (!steps_msg.data.empty()) steps_msg.data += "/";
            steps_msg.data += step.entity.target_name;
        }
        if (!steps_msg.data.empty() && steps_msg.data.front() == '/')
            steps_msg.data.erase(0, 1); // Remove the first '/'
        steps_pub_.publish(steps_msg);
        logger_.log("Published steps for exploration: " + steps_msg.data);

        // Publish the current step for exploration
        std_msgs::Int16 current_step_idx_msg;
        current_step_idx_msg.data = current_step_idx_;
        current_step_idx_pub_.publish(current_step_idx_msg);
        logger_.log("Published current step index for exploration: " + std::to_string(current_step_idx_msg.data));

        // Publish question type for exploration, path_follower
        std_msgs::String question_type_msg;
        if (current_step_idx_ <= max_step_idx_ ) {
            std::string question_type = plans_.steps[current_step_idx_].action;
            if (question_type != "find" && question_type != "count") {
                question_type = "instruction_following";
                }
                question_type_msg.data = question_type;
                question_type_pub_.publish(question_type_msg);
                logger_.log("Published question type: " + question_type_msg.data);
        }

        // Regularly check the status of the active VG node and reassign tasks if needed
        pollAndReassign();
        return;
    }
}

bool Manager::loadConfigAndComputeMaxNodes(const std::string& path){
    try {
        std::ifstream f(path);
        if(!f.is_open()){
            logger_.logerr("Failed to open config.json: " + path);
            return false;
        }
        nlohmann::json j;
        f >> j;

        // NUM_CLIENTS
        if (j.contains("NUM_CLIENTS") && j["NUM_CLIENTS"].is_number_integer()){
            num_clients_per_node_ = j["NUM_CLIENTS"].get<int>();
        } else {
            num_clients_per_node_ = 1; // default value
        }

        // GOOGLE_API_KEY
        std::ifstream ifs("/ws/external/ai_module/src/config.json");
        if (!ifs.is_open()) {
            std::cerr << "Failed to open config.json\n";
            return 1;
        }
        json data;
        ifs >> data;

        const std::string model_name = data.value("MODEL_NAME", std::string{});
        if (model_name.empty()) {
            std::cerr << "[Manager] MODEL_NAME is missing in config.json\n";
            return false;
        }
        const bool is_gemini = (model_name.find("gemini") != std::string::npos);
        const std::string api_key_name = is_gemini ? "GOOGLE_API_KEY" : "OPENAI_API_KEY";

        all_api_keys_.clear();
        for (auto it = j.begin(); it != j.end(); ++it){
            const std::string k = it.key();
            if (k.rfind(api_key_name, 0) == 0) { // prefix match
                // Check if the value is a string
                if (it.value().is_string()){
                    all_api_keys_.push_back(it.value().get<std::string>());
                }
            }
        }

        // Max VG nodes = total API keys / number of clients per node
        if (num_clients_per_node_ <= 0) num_clients_per_node_ = 1;
        max_vg_nodes_ = all_api_keys_.size() / static_cast<size_t>(num_clients_per_node_);
        logger_.log("NUM_CLIENTS=" + std::to_string(num_clients_per_node_) +
                    ", #KEYS=" + std::to_string(all_api_keys_.size()) +
                    ", max_vg_nodes=" + std::to_string(max_vg_nodes_));
        return true;
    } catch (const std::exception& e){
        logger_.logerr(std::string("Exception in loadConfigAndComputeMaxNodes: ")+ e.what());
        return false;
    }
}

size_t Manager::discoverVGNodes() {
    vg_nodes_.clear();

    int target = std::max<int>(1, (int)max_vg_nodes_);
    int min_required = 1;

    size_t found = 0;

    while (ros::ok() && found < (size_t)min_required) {
        for (int i=0; i<target; ++i) {
            if (i >= (size_t)min_required && found < (size_t)min_required) break;

            const std::string ns = node_name_grounding_ + "_" + std::to_string(i);
            auto set_cli = nh_.serviceClient<visual_grounding::SetSubplans>(ns + "/set_subplans");

            if (set_cli.waitForExistence(ros::Duration(0.5))) {
                VGNode node;
                node.ns               = ns;
                node.set_subplans_cli = set_cli;
                node.status_cli       = nh_.serviceClient<std_srvs::Trigger>(ns + "/status");
                node.reset_cli        = nh_.serviceClient<std_srvs::Trigger>(ns + "/reset");
                node.connected        = true;

                vg_nodes_.push_back(std::move(node));
                const size_t idx = vg_nodes_.size() - 1;

                // Advertise the active signal service for this VG node
                // {
                //     ros::NodeHandle nh_ns(vg_nodes_[idx].ns);
                //     vg_nodes_[idx].active_signal_srv =
                //         nh_ns.advertiseService<std_srvs::Trigger::Request, std_srvs::Trigger::Response>(
                //             "active_signal",
                //             boost::bind(&Manager::activeSignalCb, this, _1, _2, idx) 
                //         );
                // }

                ++found;
                logger_.log("[VG] connected & added: " + ns);
            }
        }
    }

    logger_.log("#connected_vg_nodes=" + std::to_string(found));
    return found;
}

void Manager::splitApiKeysPerNode(){
    // If there are no VG nodes, we cannot assign API keys
    size_t need = vg_nodes_.size() * static_cast<size_t>(num_clients_per_node_);
    if (need > all_api_keys_.size()) need = all_api_keys_.size();

    size_t idx = 0;
    for (size_t i=0; i<vg_nodes_.size(); ++i){
        vg_nodes_[i].api_keys.clear();
        for (int k=0; k<num_clients_per_node_ && idx<need; ++k, ++idx){
            vg_nodes_[i].api_keys.push_back(all_api_keys_[idx]);
        }
        // Set the API keys to the parameter server
        std::string param_name = vg_nodes_[i].ns + "/api_keys";
        nh_.setParam(param_name, vg_nodes_[i].api_keys);
        nh_.setParam(vg_nodes_[i].ns + "/num_clients", num_clients_per_node_);
        logger_.log("[VG] " + vg_nodes_[i].ns + " assigned " + std::to_string(vg_nodes_[i].api_keys.size()) + " API keys");
    }
}

bool Manager::sendSubplanToNode(VGNode& node, const task_planner::Task& current_step, WorkKind kind){
    if (!node.connected) return false;

    ros::Duration(0.5).sleep(); // Wait for the node to be ready

    logger_.log("[DEBUG] About to send subplan to " + node.ns + ", instruction size: " + std::to_string(instructions_.size()));
    
    visual_grounding::SetSubplans srv;
    srv.request.text_instruction = instructions_;
    srv.request.start_time = start_time_;

    if (kind == WorkKind::Constraint) {
        srv.request.constraints.push_back(current_step);
        srv.request.steps.clear(); // Clear steps if sending a constraint
    } else if (kind == WorkKind::Step) {
        srv.request.steps.push_back(current_step);
        srv.request.constraints.clear(); // Clear constraints if sending a step
    }

    srv.request.current_step = current_step;

    logger_.log("[DEBUG] About to call service for " + node.ns);
    try {
        if (node.set_subplans_cli.call(srv) && srv.response.success){
            logger_.log("[VG SEND] " + node.ns + " <= " + current_step.action +
                        std::string(" [") + (kind==WorkKind::Constraint?"constraint":"step") + "]");
            return true;
        } else {
            logger_.logerr("[DEBUG] Service call failed for " + node.ns);
        }
    } catch (const std::exception& e) {
        logger_.logerr("[DEBUG] Exception during service call: " + std::string(e.what()));
    }
    return false;
}

void Manager::pollAndReassign(){
    if (vg_nodes_.empty()) return;
    if (active_step_node_idx_ == -1){
        resetManager();
        logger_.log("[VG RE-ASSIGN] No active step node found. Resetting manager.");
        return;
    }

    // Check the active step node and busy VG nodes status
    logger_.log("================================================");
    std::vector<bool> done(vg_nodes_.size(), false);
    bool tmp_done = false;
    int idle_count = 0;
    for (size_t i=0; i<vg_nodes_.size(); ++i){
        if (vg_nodes_[i].connected && vg_nodes_[i].busy){
            callStatus(vg_nodes_[i], tmp_done);
            done[i] = tmp_done;

            if (vg_nodes_[i].is_active && done[i]) {
                idleVGNode(vg_nodes_[i]);
                idle_count++;
            }
        }
    }

    if (idle_count == 0) {
        return;
    }
    
    // Reassign tasks to the idle node
    if (!task_queue_.empty()){
        for (auto& node : vg_nodes_){
            if (!node.connected || node.busy) continue;
            if (task_queue_.empty()) break;

            auto at = task_queue_.front(); task_queue_.pop_front();
            if (sendSubplanToNode(node, at.task, at.kind)){
                node.busy = true;
                node.last_kind = at.kind;
                node.is_active = (at.kind == WorkKind::Constraint); // Constraint nodes are always active
                node.step_idx = at.kind == WorkKind::Step ? ++assigned_step_idx_ : -1; // Set step index only for steps
            } else {
                logger_.logerr("Failed to send subplan to " + node.ns);
            }
        }
    }

    if (done[active_step_node_idx_]) {
        // If the active step node is done, choose the next active step node
        setNextActiveStepNode();
        logger_.log("Progress: " + std::to_string(current_step_idx_) + "/" + std::to_string(max_step_idx_ + 1));
    }


    // Check if all step nodes are done
    bool any_busy = false;
    for (const auto& n : vg_nodes_) {
        if (n.last_kind != WorkKind::Constraint && n.busy) {
            any_busy = true;
            break;
        }
    }

    if (!any_busy && task_queue_.empty()) {
        logger_.log("[ALL DONE]");
        task_state_ = Done; // Set the task state to Done
    }   
}

std::string Manager::getActiveNodesString() {
    std::vector<std::string> active_nodes;
    
    for (const auto& node : vg_nodes_) {
        if (node.connected && node.is_active) {
            active_nodes.push_back(node.ns);
        }
    }
    
    // Join with "/" separator
    std::string result;
    for (size_t i = 0; i < active_nodes.size(); ++i) {
        if (i > 0) result += ",";
        result += active_nodes[i];
    }
    
    return result;
}

void Manager::publishActiveNodes() {
    std_msgs::String active_nodes_msg;
    active_nodes_msg.data = getActiveNodesString();
    active_nodes_pub_.publish(active_nodes_msg);
    logger_.log("Published active nodes: " + active_nodes_msg.data);
}

void Manager::chooseActiveFromBusyStepNodes(){
    // Initialize all step nodes as not active
    for (auto& n : vg_nodes_){
        if (n.last_kind == WorkKind::Step) {
            n.is_active = false;
        }
    }

    // Find the first busy step node and set it as active
    for (size_t i=0; i<vg_nodes_.size(); ++i){
        if (vg_nodes_[i].connected && vg_nodes_[i].busy && vg_nodes_[i].last_kind == WorkKind::Step){
            vg_nodes_[i].is_active = true;
            active_step_node_idx_ = i;
            logger_.log("[ACTIVE] " + vg_nodes_[i].ns + " (step node)");
            current_step_idx_ = vg_nodes_[i].step_idx; // Set the current step index

            // Publish the current step for exploration
            std_msgs::Int16 current_step_idx_msg;
            current_step_idx_msg.data = current_step_idx_;
            current_step_idx_pub_.publish(current_step_idx_msg);
            logger_.log("Published current step index for exploration: " + std::to_string(current_step_idx_msg.data));
            
            // Publish active nodes
            publishActiveNodes();
            return;
        }
    }

    logger_.log("[ACTIVE] No active VG nodes found.");
    publishActiveNodes(); // Publish empty active nodes
}

void Manager::setNextActiveStepNode(){
    current_step_idx_++;

    if (current_step_idx_ > max_step_idx_) {
        logger_.log("[ACTIVE] No more steps to process.");
        publishActiveNodes(); // Publish empty active nodes
        return;
    }

    for (size_t i=0; i<vg_nodes_.size(); ++i){
        if (vg_nodes_[i].step_idx == current_step_idx_){
            // Set the next node as active
            vg_nodes_[i].is_active = true;
            active_step_node_idx_ = i;
            logger_.log("[ACTIVE] " + vg_nodes_[i].ns + " (step node)");

            // Publish the current step for exploration
            std_msgs::Int16 current_step_idx_msg;
            current_step_idx_msg.data = current_step_idx_;
            current_step_idx_pub_.publish(current_step_idx_msg);
            logger_.log("Published current step index for exploration: " + std::to_string(current_step_idx_msg.data));
            
            // Publish active nodes
            publishActiveNodes();
            return;
        }
    }
}

bool Manager::callStatus(VGNode& node, bool& done){
    if (!node.connected) return false;
    std_srvs::Trigger srv;
    if (node.status_cli.call(srv)){
        std::string msg = srv.response.message; // waiting, searching, filtering, get_sanpshots, processing, completed, unknown
        
        const std::string kind   = (node.last_kind == WorkKind::Step) ? " (step," : " (constraint,";
        const std::string active = node.is_active ? " active)" : " inactive)";
        const std::string special = node.is_active && (node.last_kind == WorkKind::Step) ? "*** " + std::to_string(current_step_idx_) + "/" + std::to_string(max_step_idx_ + 1) : "";
        if (msg == "Completed") {
            done = true;
        }
        else {
            done = false;
        }
        logger_.log(std::string("[VG STATUS] ") + node.ns + " -> " + msg + kind + active + special);
        return true;
    }
    return false;
}

bool Manager::callReset(VGNode& node){
    if (!node.connected) return false;
    std_srvs::Trigger srv;
    if (node.reset_cli.call(srv)){
        logger_.log("[VG RESET] " + node.ns + " -> " + (srv.response.success ? "ok" : "fail"));
        return srv.response.success;
    }
    return false;
}

bool Manager::resetManager() {
    logger_.log("[RESET MANAGER] Resetting all states and queues...");
    task_state_ = Idle; // Reset task state to Idle
    isRunning_ = false; // Stop the running state
    hasTask_ = false; // Clear the task flag
    goal_in_progress_ = false; // Reset goal in progress flag
    read_detected_objects_ = false; // Reset read detected objects flag
    instructions_.clear(); // Clear instructions
    plans_.steps.clear(); // Clear plans
    plans_.constraints.clear(); // Clear constraints
    current_step_ = task_planner::Task(); // Reset current step
    target_pose_ = geometry_msgs::Pose2D(); // Reset target pose
    task_queue_.clear(); // Clear the task queue

    // mutex 제거 - 이미 호출하는 곳에서 걸려있음
    for (auto& node : vg_nodes_) {
        idleVGNode(node);
    }

    active_step_node_idx_ = -1; // Reset the active step node index
    vg_nodes_.clear(); // Clear the VG nodes
    connected_vg_nodes_ = 0; // Reset connected VG nodes count
    current_step_idx_ = -1; // Reset current step index
    assigned_step_idx_ = -1; // Reset assigned step index
    max_step_idx_ = -1; // Reset maximum step index

    logger_.log("[MANAGER RESET] All states and queues have been reset.");
    return true;
}

void Manager::idleVGNode(VGNode& node) {
    if (!node.connected) return;
    node.busy = false;
    node.is_active = false;
    node.step_idx = -1;
    node.last_kind = WorkKind::None;
    callReset(node);
    logger_.log("[VG IDLE] " + node.ns + " is now idle.");
    
    // Publish updated active nodes after making a node idle
    publishActiveNodes();
}

// bool Manager::activeSignalCb(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res, size_t node_index)
// {
//     if (node_index >= vg_nodes_.size()) {
//         res.success = false;
//         res.message = "invalid node index";
//         return true;
//     }
    
//     const bool ok = vg_nodes_[node_index].is_active;
//     res.success = ok;
//     res.message = ok ? "active" : "inactive";
//     logger_.log("[ACTIVE SIGNAL] " + vg_nodes_[node_index].ns + " -> " + (ok ? "active" : "inactive"));
//     return true;
// }

int main(int argc, char** argv)
{
    ros::init(argc, argv, "manager");
    ros::NodeHandle nh;

    Manager Manager(nh);

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
