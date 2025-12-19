//
// Created by dshong on 25. 12. 17.
//
#ifndef RRLOGGER_H
#define RRLOGGER_H
#include <rerun.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

// ----------------------------
// Color / Lab helpers (sRGB D65)
// ----------------------------
static double srgb_to_linear(double c_srgb);

static std::array<double, 3> rgb_to_xyz(double r, double g, double b);

static double f_lab(double t);

static std::array<double, 3> xyz_to_lab(double X, double Y, double Z);

static std::array<double, 3> rgb_to_lab(double r, double g, double b);

static double l2_dist3(const std::array<double, 3>& p, const std::array<double, 3>& q);

// ----------------------------------------
// make_palette: farthest-point sampling in Lab
// ----------------------------------------
static std::vector<rerun::components::Color> make_palette(
    int K, int n_candidates = 2000, uint32_t seed = 0
);

// ----------------------------
// RRLogger (C++)
// ----------------------------
class RRLogger {
public:
    using Value = std::variant<std::string, int64_t, double>;
    using DataMap = std::unordered_map<std::string, Value>;

    RRLogger(std::string output_path = "/ws/external/log", std::string name = "rerun_example", int max_colors = 100, bool save_to_file = true);

    // If you don't pass t_sec, you should feed your ROS time here from the caller.
    double set_time(double t_sec);

    void log(const DataMap& data, double t_sec);

    void log_single(const std::string& entity_path, const Value& value);

    const std::vector<rerun::components::Color>& palette() const { return palette_; }
    const std::string& timeline() const { return timeline_; }

private:
    std::string timeline_;
    std::vector<rerun::components::Color> palette_;
    rerun::RecordingStream rec_;
};

#endif // RRLOGGER_H