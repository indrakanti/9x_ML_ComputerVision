#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Box { double x1, y1, x2, y2; };

struct Artifact {
  int classes;
  std::vector<Box> pred_boxes;
  std::vector<std::vector<double>> pred_probs;
  std::vector<Box> gt_boxes;
  std::vector<int> gt_classes;
  double lambda_cls;
  double lambda_l1;
  double lambda_iou;
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
      std::stoi(values.at("classes")),
      {}, {}, {}, {},
      std::stod(values.at("lambda_cls")),
      std::stod(values.at("lambda_l1")),
      std::stod(values.at("lambda_iou"))};

  std::stringstream pred_boxes(values.at("pred_boxes"));
  std::string token;
  while (std::getline(pred_boxes, token, ';')) a.pred_boxes.push_back(parse_box(token));

  std::stringstream probs(values.at("pred_probs"));
  while (std::getline(probs, token, ';')) a.pred_probs.push_back(parse_values(token));

  std::stringstream gt_boxes(values.at("gt_boxes"));
  while (std::getline(gt_boxes, token, ';')) a.gt_boxes.push_back(parse_box(token));

  std::stringstream gt_classes(values.at("gt_classes"));
  while (std::getline(gt_classes, token, ',')) a.gt_classes.push_back(std::stoi(token));

  return a;
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

double l1(const Box& a, const Box& b) {
  return std::abs(a.x1 - b.x1) + std::abs(a.y1 - b.y1)
       + std::abs(a.x2 - b.x2) + std::abs(a.y2 - b.y2);
}

std::vector<std::vector<double>> costs(const Artifact& a) {
  std::vector<std::vector<double>> matrix;
  for (std::size_t p = 0; p < a.pred_boxes.size(); ++p) {
    std::vector<double> row;
    for (std::size_t g = 0; g < a.gt_boxes.size(); ++g) {
      const int cls = a.gt_classes[g];
      row.push_back(
          a.lambda_cls * (-a.pred_probs[p][static_cast<std::size_t>(cls)])
          + a.lambda_l1 * l1(a.pred_boxes[p], a.gt_boxes[g])
          + a.lambda_iou * (1.0 - iou(a.pred_boxes[p], a.gt_boxes[g])));
    }
    matrix.push_back(row);
  }
  return matrix;
}

std::vector<std::pair<int, int>> assignment(
    const std::vector<std::vector<double>>& matrix) {
  const int predictions = static_cast<int>(matrix.size());
  const int targets = static_cast<int>(matrix.front().size());

  std::vector<int> indices(predictions);
  for (int i = 0; i < predictions; ++i) indices[static_cast<std::size_t>(i)] = i;

  double best_cost = std::numeric_limits<double>::infinity();
  std::vector<std::pair<int, int>> best;

  std::sort(indices.begin(), indices.end());
  do {
    double total = 0.0;
    for (int target = 0; target < targets; ++target) {
      total += matrix[static_cast<std::size_t>(indices[static_cast<std::size_t>(target)])]
                     [static_cast<std::size_t>(target)];
    }
    if (total < best_cost) {
      best_cost = total;
      best.clear();
      for (int target = 0; target < targets; ++target) {
        best.push_back({indices[static_cast<std::size_t>(target)], target});
      }
    }
  } while (std::next_permutation(indices.begin(), indices.end()));

  return best;
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto matrix = costs(a);
  const auto matches = assignment(matrix);

  std::cout << std::setprecision(17) << "{\"costs\":[";
  for (std::size_t r = 0; r < matrix.size(); ++r) {
    if (r != 0U) std::cout << ",";
    std::cout << "[";
    for (std::size_t c = 0; c < matrix[r].size(); ++c) {
      if (c != 0U) std::cout << ",";
      std::cout << matrix[r][c];
    }
    std::cout << "]";
  }
  std::cout << "],\"assignment\":[";
  for (std::size_t i = 0; i < matches.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << "[" << matches[i].first << "," << matches[i].second << "]";
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
      return assignment(costs(load(argv[2]))).size() == 2U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
