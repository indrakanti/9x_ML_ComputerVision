#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Detection {
  double x1, y1, x2, y2, score;
  int class_id;
};

struct Artifact {
  int image_w, image_h, grid_w, grid_h, classes;
  double anchor_w, anchor_h, score_threshold, nms_iou;
  std::vector<std::vector<double>> predictions;
};

double sigmoid(double value) {
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

std::vector<double> parse_row(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

Artifact load(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open artifact");
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    const auto sep = line.find('=');
    if (sep == std::string::npos) throw std::runtime_error("invalid line");
    values[line.substr(0, sep)] = line.substr(sep + 1);
  }

  Artifact a{
      std::stoi(values.at("image_w")),
      std::stoi(values.at("image_h")),
      std::stoi(values.at("grid_w")),
      std::stoi(values.at("grid_h")),
      std::stoi(values.at("classes")),
      std::stod(values.at("anchor_w")),
      std::stod(values.at("anchor_h")),
      std::stod(values.at("score_threshold")),
      std::stod(values.at("nms_iou")),
      {}};

  std::stringstream rows(values.at("predictions"));
  std::string row;
  while (std::getline(rows, row, ';')) a.predictions.push_back(parse_row(row));
  return a;
}

double iou(const Detection& a, const Detection& b) {
  const double ix1 = std::max(a.x1, b.x1);
  const double iy1 = std::max(a.y1, b.y1);
  const double ix2 = std::min(a.x2, b.x2);
  const double iy2 = std::min(a.y2, b.y2);
  const double inter =
      std::max(0.0, ix2 - ix1) * std::max(0.0, iy2 - iy1);
  const double aa = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double ab = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aa + ab - inter);
}

std::vector<Detection> decode(const Artifact& a) {
  std::vector<Detection> out;
  for (std::size_t index = 0; index < a.predictions.size(); ++index) {
    const auto& p = a.predictions[index];
    const int gy = static_cast<int>(index) / a.grid_w;
    const int gx = static_cast<int>(index) % a.grid_w;

    int class_id = 0;
    double score = -1.0;
    for (int c = 0; c < a.classes; ++c) {
      const double candidate =
          sigmoid(p[4]) * sigmoid(p[static_cast<std::size_t>(5 + c)]);
      if (candidate > score) {
        score = candidate;
        class_id = c;
      }
    }
    if (score < a.score_threshold) continue;

    const double cx = (sigmoid(p[0]) + gx) / a.grid_w;
    const double cy = (sigmoid(p[1]) + gy) / a.grid_h;
    const double width = a.anchor_w * std::exp(p[2]) / a.image_w;
    const double height = a.anchor_h * std::exp(p[3]) / a.image_h;

    out.push_back(
        {cx - width / 2.0, cy - height / 2.0,
         cx + width / 2.0, cy + height / 2.0,
         score, class_id});
  }
  return out;
}

std::vector<Detection> nms(std::vector<Detection> detections, double threshold) {
  std::stable_sort(
      detections.begin(), detections.end(),
      [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
      });

  std::vector<Detection> kept;
  while (!detections.empty()) {
    const Detection current = detections.front();
    kept.push_back(current);
    std::vector<Detection> next;
    for (std::size_t i = 1; i < detections.size(); ++i) {
      const auto& candidate = detections[i];
      if (candidate.class_id != current.class_id ||
          iou(current, candidate) <= threshold) {
        next.push_back(candidate);
      }
    }
    detections = std::move(next);
  }
  return kept;
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto detections = nms(decode(a), a.nms_iou);
  std::cout << std::setprecision(17) << "{\"detections\":[";
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (i != 0U) std::cout << ",";
    const auto& d = detections[i];
    std::cout << "{\"x1\":" << d.x1
              << ",\"y1\":" << d.y1
              << ",\"x2\":" << d.x2
              << ",\"y2\":" << d.y2
              << ",\"score\":" << d.score
              << ",\"class_id\":" << d.class_id << "}";
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      const Artifact a = load(argv[2]);
      return nms(decode(a), a.nms_iou).empty()
                 ? EXIT_FAILURE
                 : EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
