// logger.h
#pragma once

#include <ros/ros.h>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

class Logger {
public:
    Logger() : quiet_(false) {}

    explicit Logger(bool quiet, const std::string& prefix, const std::string& log_path)
        : quiet_(quiet), prefix_(prefix), log_path_(log_path) {
        prepareDirIfNeeded(log_path_);
        announce();
    }

    void setQuiet(bool quiet) { quiet_ = quiet; }
    void setPrefix(const std::string& prefix) { prefix_ = prefix; }
    void setLogPath(const std::string& log_path) {
        log_path_ = log_path;
        prepareDirIfNeeded(log_path_);
    }

    void configure(bool quiet, const std::string& prefix, const std::string& log_path) {
        quiet_ = quiet;
        prefix_ = prefix;
        log_path_ = log_path;
        prepareDirIfNeeded(log_path_);
        std::ofstream(log_path_, std::ios::trunc).close();
        announce();
    }

    void logerr(const std::string& msg) { log(msg, "error"); }
    void logwarn(const std::string& msg) { log(msg, "warn"); }
    void loginfo(const std::string& msg) { log(msg, "info"); }
    void logdebug(const std::string& msg) { log(msg, "debug"); }

    void log(const std::string& msg, const std::string& level = "")
    {
        // 파일 저장
        if (!log_path_.empty()) {
            std::ofstream file(log_path_, std::ios_base::app);
            if (file.is_open()) {
                file << "[" << nowString() << "] " << msg << "\n";
                file.close();
            }
        }

        if (!quiet_) {
            std::string full_msg = prefix_.empty() ? msg : "[" + prefix_ + "] " + msg;

            if (level == "warn") {
                ROS_WARN("%s", full_msg.c_str());
            } else if (level == "error") {
                ROS_ERROR("%s", full_msg.c_str());
            } else if (level == "debug") {
                ROS_DEBUG("%s", full_msg.c_str());
            } else if (level == "info") {
                ROS_INFO("%s", full_msg.c_str());
            } else {
                std::cout << full_msg << std::endl;
            }
        }
    }

private:
    bool quiet_;
    std::string prefix_;
    std::string log_path_;

    static void prepareDirIfNeeded(const std::string& log_path) {
        if (log_path.empty()) return;
        size_t pos = log_path.find_last_of('/');
        if (pos != std::string::npos) {
            std::string dir = log_path.substr(0, pos);
            mkdir(dir.c_str(), 0755);
        }
    }

    void announce() const {
        const std::string mode_str = quiet_ ? "quiet" : "verbose";
        std::cout << "[Logger] " << (prefix_.empty() ? "Logger" : prefix_)
                  << " is set to " << mode_str << " mode!" << std::endl;
    }

    static std::string nowString() {
        auto t = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};
