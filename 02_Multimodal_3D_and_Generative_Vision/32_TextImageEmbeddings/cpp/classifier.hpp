#pragma once

#include <memory>
#include <string>
#include <vector>

namespace cv9x::module32 {
struct RankedClass {
    std::string label;
    float probability;
};

// Reusable C++17 application interface: image path → ranked CIFAR-10 predictions.
// Model loading and checksum/manifest validation occur at construction.
class Classifier {
public:
    explicit Classifier(const std::string& export_directory);
    ~Classifier();
    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;
    Classifier(Classifier&&) noexcept;
    Classifier& operator=(Classifier&&) noexcept;
    std::vector<RankedClass> predict(const std::string& image_path) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace cv9x::module32
