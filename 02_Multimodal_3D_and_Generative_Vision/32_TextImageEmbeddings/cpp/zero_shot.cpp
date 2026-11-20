#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Embedding = std::vector<double>;

Embedding normalize(const Embedding& input) {
    double squared = 0.0;
    for (const auto value : input) squared += value * value;
    const double norm = std::sqrt(squared);
    if (!std::isfinite(norm) || norm <= 1e-12) throw std::invalid_argument("invalid embedding norm");
    Embedding result;
    for (const auto value : input) result.push_back(value / norm);
    return result;
}

double score(const Embedding& image, const Embedding& text, double temperature = 1.0) {
    if (!std::isfinite(temperature) || temperature <= 0.0)
        throw std::invalid_argument("invalid temperature");
    if (image.size() != text.size() || image.empty())
        throw std::invalid_argument("embedding dimension mismatch");
    const auto a = normalize(image);
    const auto b = normalize(text);
    double dot = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot / temperature;
}

int main() {
    const Embedding image{3.0, 4.0};
    const std::vector<Embedding> texts{{3.0, 4.0}, {4.0, -3.0}, {-3.0, -4.0}};
    const std::vector<double> expected{1.0, 0.0, -1.0};
    for (std::size_t i = 0; i < texts.size(); ++i) {
        const auto value = score(image, texts[i]);
        std::cout << std::fixed << std::setprecision(6) << value << '\n';
        if (std::abs(value - expected[i]) > 1e-9) return 1;
    }
    return 0;
}
