#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Vector = std::vector<double>;

Vector softmax(const Vector& logits) {
    if (logits.empty()) throw std::invalid_argument("empty logits");
    for (const auto value : logits)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite logit");
    double peak = logits.front();
    for (const auto value : logits) if (value > peak) peak = value;
    Vector weights;
    double total = 0.0;
    for (const auto value : logits) {
        const double weight = std::exp(value - peak);
        weights.push_back(weight);
        total += weight;
    }
    for (auto& weight : weights) weight /= total;
    return weights;
}

int main() {
    const Vector query{1.0, 0.0};
    const std::vector<Vector> patches{{1.0, 0.0}, {0.0, 1.0}};
    Vector logits;
    for (const auto& patch : patches) {
        double dot = 0.0;
        for (std::size_t i = 0; i < query.size(); ++i) dot += query[i] * patch[i];
        logits.push_back(dot / std::sqrt(static_cast<double>(query.size())));
    }
    const Vector weights = softmax(logits);
    Vector context(query.size(), 0.0);
    for (std::size_t j = 0; j < patches.size(); ++j)
        for (std::size_t i = 0; i < query.size(); ++i)
            context[i] += weights[j] * patches[j][i];
    if (!(weights[0] > weights[1]) || std::abs(weights[0] + weights[1] - 1.0) > 1e-12)
        return 1;
    std::cout << std::fixed << std::setprecision(9);
    for (const auto value : weights) std::cout << value << '\n';
    for (const auto value : context) std::cout << value << '\n';
    return 0;
}
