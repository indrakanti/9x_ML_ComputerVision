#include "classifier.hpp"

#include <nlohmann/json.hpp>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * p / 100.0;
    const auto index = static_cast<std::size_t>(position);
    const double fraction = position - static_cast<double>(index);
    if (index + 1 == values.size()) return values.back();
    return values[index] * (1.0 - fraction) + values[index + 1] * fraction;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string directory;
        std::string image;
        int warmup = 3;
        int repeats = 10;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("option missing value");
            const std::string option = argv[index];
            const std::string value = argv[index + 1];
            if (option == "--export-dir") directory = value;
            else if (option == "--image") image = value;
            else if (option == "--warmup") warmup = std::stoi(value);
            else if (option == "--repeats") repeats = std::stoi(value);
            else throw std::invalid_argument("unknown option: " + option);
        }
        if (directory.empty() || image.empty() || warmup < 0 ||
            repeats < 1 || repeats > 10000 || warmup > 10000)
            throw std::invalid_argument(
                "usage: cv9x_module32_classify --export-dir DIR --image FILE "
                "[--warmup 3] [--repeats 10]");
        cv9x::module32::Classifier classifier(directory);
        for (int i = 0; i < warmup; ++i) (void)classifier.predict(image);
        std::vector<double> elapsed;
        std::vector<cv9x::module32::RankedClass> results;
        for (int i = 0; i < repeats; ++i) {
            const auto start = std::chrono::steady_clock::now();
            results = classifier.predict(image);
            const auto finish = std::chrono::steady_clock::now();
            elapsed.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
        }
        nlohmann::json output;
        output["prediction"] = results.front().label;
        output["ranked"] = nlohmann::json::array();
        for (const auto& result : results)
            output["ranked"].push_back({
                {"label", result.label}, {"probability", result.probability},
            });
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("cannot read process resource usage");
        output["benchmark"] = {
            {"backend", "onnxruntime_cpu_native"},
            {"scope", "image decode, preprocess, image encoder and scoring; "
                      "text encoder cached; sessions loaded before timing"},
            {"warmup", warmup},
            {"repeats", repeats},
            {"latency_p50_ms", percentile(elapsed, 50.0)},
            {"latency_p95_ms", percentile(elapsed, 95.0)},
            {"latency_mean_ms", std::accumulate(elapsed.begin(), elapsed.end(), 0.0) /
                                    static_cast<double>(repeats)},
            {"peak_process_rss_kib", usage.ru_maxrss},
        };
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Module 32 native classification failed: " << error.what() << '\n';
        return 1;
    }
}
