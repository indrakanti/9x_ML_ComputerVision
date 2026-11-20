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
#include <vector>

namespace {

struct Keypoint { double x, y, confidence; };

struct Artifact {
  int heatmap_h;
  int heatmap_w;
  int image_h;
  int image_w;
  int keypoints;
  std::vector<std::vector<double>> heatmaps;
  std::vector<std::vector<double>> ground_truth;
  std::vector<int> visible;
  double normalization;
  double pck_alpha;
};

std::vector<double> parse_doubles(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

std::vector<int> parse_ints(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stoi(token));
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
      std::stoi(values.at("heatmap_h")),
      std::stoi(values.at("heatmap_w")),
      std::stoi(values.at("image_h")),
      std::stoi(values.at("image_w")),
      std::stoi(values.at("keypoints")),
      {}, {}, parse_ints(values.at("visible")),
      std::stod(values.at("normalization")),
      std::stod(values.at("pck_alpha"))};

  std::stringstream heatmaps(values.at("heatmaps"));
  std::string token;
  while (std::getline(heatmaps, token, ';')) {
    a.heatmaps.push_back(parse_doubles(token));
  }

  std::stringstream truth(values.at("ground_truth"));
  while (std::getline(truth, token, ';')) {
    a.ground_truth.push_back(parse_doubles(token));
  }
  return a;
}

std::vector<Keypoint> decode(const Artifact& a) {
  const double sx = static_cast<double>(a.image_w) / a.heatmap_w;
  const double sy = static_cast<double>(a.image_h) / a.heatmap_h;
  std::vector<Keypoint> output;

  for (const auto& heatmap : a.heatmaps) {
    const auto best = std::max_element(heatmap.begin(), heatmap.end());
    const int index = static_cast<int>(std::distance(heatmap.begin(), best));
    const int y = index / a.heatmap_w;
    const int x = index % a.heatmap_w;
    output.push_back(
        {(x + 0.5) * sx - 0.5,
         (y + 0.5) * sy - 0.5,
         *best});
  }
  return output;
}

double pck(const Artifact& a, const std::vector<Keypoint>& predicted) {
  const double threshold = a.pck_alpha * a.normalization;
  int correct = 0;
  int valid = 0;

  for (int i = 0; i < a.keypoints; ++i) {
    if (a.visible[static_cast<std::size_t>(i)] == 0) continue;
    ++valid;
    const double dx =
        predicted[static_cast<std::size_t>(i)].x
        - a.ground_truth[static_cast<std::size_t>(i)][0];
    const double dy =
        predicted[static_cast<std::size_t>(i)].y
        - a.ground_truth[static_cast<std::size_t>(i)][1];
    if (std::hypot(dx, dy) <= threshold) ++correct;
  }
  return static_cast<double>(correct) / valid;
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto points = decode(a);

  std::cout << std::setprecision(17) << "{\"keypoints\":[";
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << "{\"x\":" << points[i].x
              << ",\"y\":" << points[i].y
              << ",\"confidence\":" << points[i].confidence
              << "}";
  }
  std::cout << "],\"pck\":" << pck(a, points) << "}\n";
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
      return decode(a).size() == static_cast<std::size_t>(a.keypoints)
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
