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
#include <vector>

namespace {

struct Artifact {
  int classes;
  int height;
  int width;
  int ignore_label;
  std::vector<int> ground_truth;
  std::vector<std::vector<double>> logits;
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
      std::stoi(values.at("classes")),
      std::stoi(values.at("height")),
      std::stoi(values.at("width")),
      std::stoi(values.at("ignore_label")),
      parse_ints(values.at("ground_truth")),
      {}};

  std::stringstream rows(values.at("logits"));
  std::string token;
  while (std::getline(rows, token, ';')) {
    a.logits.push_back(parse_doubles(token));
  }
  return a;
}

std::vector<int> predictions(const Artifact& a) {
  std::vector<int> output;
  for (const auto& row : a.logits) {
    const auto best = std::max_element(row.begin(), row.end());
    output.push_back(static_cast<int>(std::distance(row.begin(), best)));
  }
  return output;
}

std::vector<std::vector<int>> confusion(
    const Artifact& a,
    const std::vector<int>& pred) {
  std::vector<std::vector<int>> matrix(
      static_cast<std::size_t>(a.classes),
      std::vector<int>(static_cast<std::size_t>(a.classes), 0));

  for (std::size_t i = 0; i < pred.size(); ++i) {
    const int truth = a.ground_truth[i];
    if (truth == a.ignore_label) continue;
    matrix[static_cast<std::size_t>(truth)]
          [static_cast<std::size_t>(pred[i])] += 1;
  }
  return matrix;
}

double class_iou(const std::vector<std::vector<int>>& matrix, int cls) {
  const int tp = matrix[static_cast<std::size_t>(cls)]
                       [static_cast<std::size_t>(cls)];
  int fp = 0;
  int fn = 0;
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    if (static_cast<int>(row) != cls) {
      fp += matrix[row][static_cast<std::size_t>(cls)];
    }
  }
  for (std::size_t col = 0; col < matrix.size(); ++col) {
    if (static_cast<int>(col) != cls) {
      fn += matrix[static_cast<std::size_t>(cls)][col];
    }
  }
  const int denominator = tp + fp + fn;
  if (denominator == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(tp) / denominator;
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto pred = predictions(a);
  const auto matrix = confusion(a, pred);

  std::vector<double> ious;
  double sum = 0.0;
  int valid = 0;
  for (int cls = 0; cls < a.classes; ++cls) {
    const double value = class_iou(matrix, cls);
    ious.push_back(value);
    if (!std::isnan(value)) {
      sum += value;
      ++valid;
    }
  }
  const double miou = sum / valid;

  std::cout << std::setprecision(17) << "{\"predictions\":[";
  for (std::size_t i = 0; i < pred.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << pred[i];
  }
  std::cout << "],\"confusion\":[";
  for (std::size_t r = 0; r < matrix.size(); ++r) {
    if (r != 0U) std::cout << ",";
    std::cout << "[";
    for (std::size_t c = 0; c < matrix[r].size(); ++c) {
      if (c != 0U) std::cout << ",";
      std::cout << matrix[r][c];
    }
    std::cout << "]";
  }
  std::cout << "],\"iou\":[";
  for (std::size_t i = 0; i < ious.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << ious[i];
  }
  std::cout << "],\"miou\":" << miou << "}\n";
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
      return predictions(a).size() ==
                     static_cast<std::size_t>(a.height * a.width)
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
