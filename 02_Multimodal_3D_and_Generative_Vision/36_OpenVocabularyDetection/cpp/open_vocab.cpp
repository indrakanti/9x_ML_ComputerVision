#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Vector = std::vector<double>;
struct Box { double x1, y1, x2, y2; };
struct Detection { int box, label; double score; };

double cosine(const Vector& a, const Vector& b) {
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("embedding size");
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    if (aa <= 1e-24 || bb <= 1e-24) throw std::invalid_argument("zero norm");
    return dot / std::sqrt(aa * bb);
}

double iou(const Box& a, const Box& b) {
    const double width = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const double height = std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return width * height / (area_a + area_b - width * height);
}

int main() {
    const std::vector<Box> boxes{{0, 0, 2, 2}, {0.2, 0.2, 2.2, 2.2}, {4, 4, 6, 6}};
    const std::vector<Vector> regions{{1, 0}, {0.8, 0.2}, {0, 1}};
    const std::vector<Vector> texts{{1, 0}, {0, 1}};
    std::vector<Detection> candidates;
    for (std::size_t i = 0; i < regions.size(); ++i) {
        int best = 0;
        double score = cosine(regions[i], texts[0]);
        for (std::size_t j = 1; j < texts.size(); ++j) {
            const double next = cosine(regions[i], texts[j]);
            if (next > score) { score = next; best = static_cast<int>(j); }
        }
        if (score >= 0.5) candidates.push_back({static_cast<int>(i), best, score});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.box < b.box;
    });
    std::vector<Detection> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& prior : kept)
            if (candidate.label == prior.label &&
                iou(boxes[static_cast<std::size_t>(candidate.box)],
                    boxes[static_cast<std::size_t>(prior.box)]) > 0.5)
                suppressed = true;
        if (!suppressed) kept.push_back(candidate);
    }
    if (kept.size() != 2 || kept[0].box != 0 || kept[1].box != 2) return 1;
    std::cout << std::fixed << std::setprecision(9);
    for (const auto& d : kept)
        std::cout << d.box << ' ' << d.label << ' ' << d.score << '\n';
    return 0;
}
