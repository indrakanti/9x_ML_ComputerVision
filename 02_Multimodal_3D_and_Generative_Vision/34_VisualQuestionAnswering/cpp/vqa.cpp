#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::vector<double>;

Vector softmax(const Vector& logits) {
    if (logits.empty()) throw std::invalid_argument("empty logits");
    for (double value : logits)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite logits");
    double peak = logits.front();
    for (double value : logits) if (value > peak) peak = value;
    Vector weights;
    double sum = 0.0;
    for (double value : logits) {
        const double weight = std::exp(value - peak);
        weights.push_back(weight);
        sum += weight;
    }
    for (double& weight : weights) weight /= sum;
    return weights;
}

int main() {
    const Vector question{1.0, 0.0};
    const std::vector<Vector> patches{{1.0, 0.0}, {0.0, 1.0}};
    const std::vector<std::string> labels{"red car", "blue bus"};
    Vector logits;
    for (const auto& patch : patches) {
        double dot = 0.0;
        for (std::size_t i = 0; i < question.size(); ++i)
            dot += question[i] * patch[i];
        logits.push_back(dot / std::sqrt(static_cast<double>(question.size())));
    }
    const Vector attention = softmax(logits);
    std::size_t best = 0;
    for (std::size_t i = 1; i < attention.size(); ++i)
        if (attention[i] > attention[best]) best = i;
    if (labels[best] != "red car" ||
        std::abs(attention[0] + attention[1] - 1.0) > 1e-12) return 1;
    std::cout << labels[best] << '\n' << std::fixed << std::setprecision(9);
    for (double weight : attention) std::cout << weight << '\n';
    return 0;
}
