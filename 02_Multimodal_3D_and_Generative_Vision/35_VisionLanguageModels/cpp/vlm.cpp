#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

Matrix project(const Matrix& patches, const Matrix& weights) {
    if (patches.empty() || weights.empty() || weights.front().empty())
        throw std::invalid_argument("empty projection");
    const std::size_t input = weights.size();
    const std::size_t output = weights.front().size();
    for (const auto& row : weights)
        if (row.size() != output) throw std::invalid_argument("ragged weights");
    Matrix result;
    for (const auto& patch : patches) {
        if (patch.size() != input) throw std::invalid_argument("width mismatch");
        Vector token(output, 0.0);
        for (std::size_t j = 0; j < output; ++j)
            for (std::size_t k = 0; k < input; ++k)
                token[j] += patch[k] * weights[k][j];
        result.push_back(token);
    }
    return result;
}

int main() {
    const Matrix patches{{1.0, 2.0}, {3.0, 4.0}};
    const Matrix weights{{1.0, 0.5, 0.0}, {0.0, 0.5, 1.0}};
    const Matrix text{{9.0, 9.0, 9.0}, {0.0, 0.0, 0.0}, {8.0, 8.0, 8.0}};
    const Matrix visual = project(patches, weights);
    Matrix fused{text.front()};
    fused.insert(fused.end(), visual.begin(), visual.end());
    fused.push_back(text.back());
    if (fused.size() != 4 || std::abs(fused[1][1] - 1.5) > 1e-12) return 1;
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& token : fused) {
        for (std::size_t j = 0; j < token.size(); ++j) {
            if (j) std::cout << ' ';
            std::cout << token[j];
        }
        std::cout << '\n';
    }
    return 0;
}
