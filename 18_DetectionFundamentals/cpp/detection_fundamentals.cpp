#include <algorithm>
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

struct Box { double x1, y1, x2, y2; };

Box parse_box(const std::string& text) {
  std::stringstream stream(text);
  std::string token;
  std::vector<double> values;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  if (values.size() != 4U) throw std::runtime_error("invalid box");
  return {values[0], values[1], values[2], values[3]};
}

double area(const Box& b) {
  if (b.x2 <= b.x1 || b.y2 <= b.y1) throw std::runtime_error("invalid box");
  return (b.x2 - b.x1) * (b.y2 - b.y1);
}

double iou(const Box& a, const Box& b) {
  const double ix1 = std::max(a.x1, b.x1);
  const double iy1 = std::max(a.y1, b.y1);
  const double ix2 = std::min(a.x2, b.x2);
  const double iy2 = std::min(a.y2, b.y2);
  const double inter =
      std::max(0.0, ix2 - ix1) * std::max(0.0, iy2 - iy1);
  return inter / (area(a) + area(b) - inter);
}

struct Artifact {
  Box a, b;
  std::vector<Box> boxes;
  std::vector<double> scores;
  std::vector<int> classes;
  double threshold;
};

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

  Artifact out{parse_box(values.at("iou_a")),
               parse_box(values.at("iou_b")),
               {}, {}, {}, std::stod(values.at("nms_iou"))};

  std::stringstream boxes(values.at("boxes"));
  std::string box;
  while (std::getline(boxes, box, ';')) out.boxes.push_back(parse_box(box));

  std::stringstream scores(values.at("scores"));
  std::string token;
  while (std::getline(scores, token, ',')) out.scores.push_back(std::stod(token));

  std::stringstream classes(values.at("classes"));
  while (std::getline(classes, token, ',')) out.classes.push_back(std::stoi(token));

  if (out.boxes.size() != out.scores.size() ||
      out.boxes.size() != out.classes.size()) {
    throw std::runtime_error("array size mismatch");
  }
  return out;
}

std::vector<int> nms(const Artifact& a) {
  std::vector<int> order(a.boxes.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return a.scores[static_cast<std::size_t>(lhs)] >
           a.scores[static_cast<std::size_t>(rhs)];
  });

  std::vector<int> keep;
  while (!order.empty()) {
    const int current = order.front();
    keep.push_back(current);
    std::vector<int> next;
    for (std::size_t i = 1; i < order.size(); ++i) {
      const int candidate = order[i];
      if (a.classes[static_cast<std::size_t>(candidate)] !=
              a.classes[static_cast<std::size_t>(current)] ||
          iou(a.boxes[static_cast<std::size_t>(current)],
              a.boxes[static_cast<std::size_t>(candidate)]) <= a.threshold) {
        next.push_back(candidate);
      }
    }
    order = std::move(next);
  }
  return keep;
}

void print_json(const std::string& path) {
  const auto artifact = load(path);
  const auto keep = nms(artifact);
  std::cout << std::setprecision(17)
            << "{\"iou\":" << iou(artifact.a, artifact.b)
            << ",\"keep\":[";
  for (std::size_t i = 0; i < keep.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << keep[i];
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
      return nms(load(argv[2])).size() == 2U ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
