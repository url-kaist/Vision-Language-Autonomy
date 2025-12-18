#include "rr_logger.h"

namespace fs = std::filesystem;

// ----------------------------
// Color / Lab helpers (sRGB D65)
// ----------------------------
static double srgb_to_linear(double c_srgb) {
    // c_srgb in [0,1]
    return (c_srgb <= 0.04045) ? (c_srgb / 12.92) : std::pow((c_srgb + 0.055) / 1.055, 2.4);
}

static std::array<double, 3> rgb_to_xyz(double r, double g, double b) {
    // r,g,b in [0,1] sRGB
    r = srgb_to_linear(r);
    g = srgb_to_linear(g);
    b = srgb_to_linear(b);

    // sRGB -> XYZ (D65)
    const double X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    const double Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    const double Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    return {X, Y, Z};
}

static double f_lab(double t) {
    // CIE Lab f(t)
    constexpr double delta = 6.0 / 29.0;
    constexpr double delta3 = delta * delta * delta;
    return (t > delta3) ? std::cbrt(t) : (t / (3.0 * delta * delta) + 4.0 / 29.0);
}

static std::array<double, 3> xyz_to_lab(double X, double Y, double Z) {
    // D65 reference white
    constexpr double Xn = 0.95047;
    constexpr double Yn = 1.00000;
    constexpr double Zn = 1.08883;

    const double fx = f_lab(X / Xn);
    const double fy = f_lab(Y / Yn);
    const double fz = f_lab(Z / Zn);

    const double L = 116.0 * fy - 16.0;
    const double a = 500.0 * (fx - fy);
    const double b = 200.0 * (fy - fz);
    return {L, a, b};
}

static std::array<double, 3> rgb_to_lab(double r, double g, double b) {
    const auto xyz = rgb_to_xyz(r, g, b);
    return xyz_to_lab(xyz[0], xyz[1], xyz[2]);
}

static double l2_dist3(const std::array<double, 3>& p, const std::array<double, 3>& q) {
    const double dx = p[0] - q[0];
    const double dy = p[1] - q[1];
    const double dz = p[2] - q[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ----------------------------------------
// make_palette: farthest-point sampling in Lab
// ----------------------------------------
static std::vector<rerun::components::Color> make_palette(
    int K, int n_candidates, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.1, 0.9);

    struct Candidate {
        double r, g, b;                 // [0,1]
        std::array<double, 3> lab;      // Lab
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(n_candidates));

    for (int i = 0; i < n_candidates; ++i) {
        Candidate c;
        c.r = unif(rng);
        c.g = unif(rng);
        c.b = unif(rng);
        c.lab = rgb_to_lab(c.r, c.g, c.b);
        candidates.push_back(c);
    }

    if (K <= 0) return {};
    K = std::min(K, n_candidates);

    std::uniform_int_distribution<int> idx_unif(0, n_candidates - 1);
    int idx = idx_unif(rng);

    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(K));
    selected.push_back(idx);

    // dist[i] = min distance to any selected point (in Lab space)
    std::vector<double> dist(static_cast<size_t>(n_candidates), std::numeric_limits<double>::infinity());
    for (int i = 0; i < n_candidates; ++i) {
        dist[static_cast<size_t>(i)] = l2_dist3(candidates[i].lab, candidates[idx].lab);
    }

    for (int k = 1; k < K; ++k) {
        // pick farthest
        int best_i = 0;
        double best_d = -1.0;
        for (int i = 0; i < n_candidates; ++i) {
            const double d = dist[static_cast<size_t>(i)];
            if (d > best_d) {
                best_d = d;
                best_i = i;
            }
        }
        selected.push_back(best_i);

        // update min distances
        for (int i = 0; i < n_candidates; ++i) {
            const double d = l2_dist3(candidates[i].lab, candidates[best_i].lab);
            auto& cur = dist[static_cast<size_t>(i)];
            if (d < cur) cur = d;
        }
    }

    std::vector<rerun::components::Color> palette;
    palette.reserve(static_cast<size_t>(K));
    for (int si : selected) {
        const auto& c = candidates[si];
        const uint8_t R = static_cast<uint8_t>(std::lround(c.r * 255.0));
        const uint8_t G = static_cast<uint8_t>(std::lround(c.g * 255.0));
        const uint8_t B = static_cast<uint8_t>(std::lround(c.b * 255.0));
        palette.emplace_back(R, G, B, 255); // Color(r,g,b,a) :contentReference[oaicite:4]{index=4}
    }
    return palette;
}

// ----------------------------
// RRLogger (C++)
// ----------------------------
RRLogger::RRLogger(
    std::string output_path, std::string name, int max_colors, bool save_to_file
) : timeline_("ros_time"), palette_(make_palette(max_colors)), rec_(std::move(name))
{
    fs::create_directories(output_path);

    const auto full_output_path = fs::path(output_path) / "test_logger.rrd";

    if (save_to_file) {
        rec_.save(full_output_path.string()).exit_on_failure();  // stream to .rrd :contentReference[oaicite:5]{index=5}
    } else {
        rec_.spawn().exit_on_failure();  // spawn viewer :contentReference[oaicite:6]{index=6}
    }
}

// If you don't pass t_sec, you should feed your ROS time here from the caller.
double RRLogger::set_time(double t_sec) {
    rec_.set_time_seconds(timeline_, t_sec); // timeline time (thread-local) :contentReference[oaicite:7]{index=7}
    return t_sec;
}

void RRLogger::log(const DataMap& data, double t_sec) {
    set_time(t_sec);
    for (const auto& kv : data) {
        log_single(kv.first, kv.second);
    }
}

void RRLogger::log_single(const std::string& entity_path, const Value& value) {
    std::visit(
        [&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                rec_.log(entity_path, rerun::archetypes::TextLog(v)); // :contentReference[oaicite:8]{index=8}
            } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                rec_.log(entity_path, rerun::archetypes::Scalar(static_cast<double>(v))); // :contentReference[oaicite:9]{index=9}
            }
        },
        value
    );
}

// ----------------------------
// Example main
// ----------------------------
int main() {
    // save_to_file=true: 기록을 /ws/external/log/test_logger.rrd 로 저장
    // save_to_file=false: viewer spawn 후 스트리밍
    RRLogger logger("/ws/external/log", "rerun_example", 100, /*save_to_file=*/true);

    // 예시: 현재 시간을 외부에서 넣어준다고 가정 (ROS라면 ros::Time::now().toSec())
    const double t_sec = 1234.567;

    RRLogger::DataMap data;
    data["status/Status"] = std::string("OK");
    data["status/#InferenceQueue"] = int64_t(7);
    data["status/AnswerScore"] = 0.93;

    logger.log(data, t_sec);

    std::cout << "Logged one timestep at t=" << t_sec << "\n";
    std::cout << "Open with: rerun /ws/external/log/test_logger.rrd\n";
    return 0;
}
