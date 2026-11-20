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

struct Box { double x1, y1, x2, y2; };

struct Artifact {
  int image_w, image_h;
  std::vector<Box> anchors;
  std::vector<std::vector<double>> deltas;
  std::vector<double> objectness;
  int pre_nms_topk;
  int post_nms_topk;
  double nms_iou;
};

std::vector<double> parse_values(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

Box parse_box(const std::string& text) {
  const auto values = parse_values(text);
  if (values.size() != 4U) throw std::runtime_error("invalid box");
  return {values[0], values[1], values[2], values[3]};
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
      {}, {}, {},
      std::stoi(values.at("pre_nms_topk")),
      std::stoi(values.at("post_nms_topk")),
      std::stod(values.at("nms_iou"))};

  std::stringstream anchors(values.at("anchors"));
  std::string token;
  while (std::getline(anchors, token, ';')) a.anchors.push_back(parse_box(token));

  std::stringstream deltas(values.at("deltas"));
  while (std::getline(deltas, token, ';')) a.deltas.push_back(parse_values(token));

  a.objectness = parse_values(values.at("objectness"));
  if (a.anchors.size() != a.deltas.size() ||
      a.anchors.size() != a.objectness.size()) {
    throw std::runtime_error("proposal array mismatch");
  }
  return a;
}

Box decode(const Box& anchor, const std::vector<double>& delta) {
  const double width = anchor.x2 - anchor.x1;
  const double height = anchor.y2 - anchor.y1;
  const double cx = (anchor.x1 + anchor.x2) / 2.0;
  const double cy = (anchor.y1 + anchor.y2) / 2.0;

  const double px = cx + delta[0] * width;
  const double py = cy + delta[1] * height;
  const double pw = width * std::exp(delta[2]);
  const double ph = height * std::exp(delta[3]);
  return {px - pw / 2.0, py - ph / 2.0,
          px + pw / 2.0, py + ph / 2.0};
}

Box clip(Box box, int width, int height) {
  box.x1 = std::max(0.0, std::min(static_cast<double>(width), box.x1));
  box.y1 = std::max(0.0, std::min(static_cast<double>(height), box.y1));
  box.x2 = std::max(0.0, std::min(static_cast<double>(width), box.x2));
  box.y2 = std::max(0.0, std::min(static_cast<double>(height), box.y2));
  return box;
}

double iou(const Box& a, const Box& b) {
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

std::vector<std::pair<Box, double>> proposals(const Artifact& a) {
  std::vector<std::pair<Box, double>> candidates;
  for (std::size_t i = 0; i < a.anchors.size(); ++i) {
    candidates.push_back(
        {clip(decode(a.anchors[i], a.deltas[i]), a.image_w, a.image_h),
         a.objectness[i]});
  }

  std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
      });
  if (candidates.size() > static_cast<std::size_t>(a.pre_nms_topk)) {
    candidates.resize(static_cast<std::size_t>(a.pre_nms_topk));
  }

  std::vector<std::pair<Box, double>> kept;
  while (!candidates.empty() &&
         kept.size() < static_cast<std::size_t>(a.post_nms_topk)) {
    const auto current = candidates.front();
    kept.push_back(current);
    std::vector<std::pair<Box, double>> next;
    for (std::size_t i = 1; i < candidates.size(); ++i) {
      if (iou(current.first, candidates[i].first) <= a.nms_iou) {
        next.push_back(candidates[i]);
      }
    }
    candidates = std::move(next);
  }
  return kept;
}

void print_json(const std::string& path) {
  const auto output = proposals(load(path));
  std::cout << std::setprecision(17) << "{\"proposals\":[";
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (i != 0U) std::cout << ",";
    const auto& box = output[i].first;
    std::cout << "{\"x1\":" << box.x1
              << ",\"y1\":" << box.y1
              << ",\"x2\":" << box.x2
              << ",\"y2\":" << box.y2
              << ",\"score\":" << output[i].second << "}";
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
      return proposals(load(argv[2])).size() == 2U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
