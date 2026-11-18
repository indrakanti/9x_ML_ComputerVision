#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Point { int x, y; };

struct Artifact {
  int height;
  int width;
  double threshold;
  std::vector<Point> positive;
  std::vector<Point> negative;
  std::vector<double> scores;
  std::vector<std::vector<double>> masks;
};

std::vector<double> parse_doubles(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

std::vector<Point> parse_points(const std::string& text) {
  std::vector<Point> points;
  if (text.empty()) return points;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ';')) {
    std::stringstream pair(item);
    std::string x;
    std::string y;
    std::getline(pair, x, ',');
    std::getline(pair, y, ',');
    points.push_back({std::stoi(x), std::stoi(y)});
  }
  return points;
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
      std::stoi(values.at("height")),
      std::stoi(values.at("width")),
      std::stod(values.at("threshold")),
      parse_points(values.at("positive")),
      parse_points(values.at("negative")),
      parse_doubles(values.at("scores")),
      {}};

  std::stringstream rows(values.at("masks"));
  std::string token;
  while (std::getline(rows, token, ';')) {
    a.masks.push_back(parse_doubles(token));
  }
  return a;
}

bool consistent(
    const Artifact& a,
    const std::vector<int>& mask) {
  const auto value_at = [&](const Point& point) {
    return mask[static_cast<std::size_t>(point.y * a.width + point.x)];
  };

  for (const auto& point : a.positive) {
    if (value_at(point) != 1) return false;
  }
  for (const auto& point : a.negative) {
    if (value_at(point) != 0) return false;
  }
  return true;
}

std::pair<int, std::vector<int>> select(const Artifact& a) {
  std::vector<int> order(a.scores.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }

  std::stable_sort(
      order.begin(),
      order.end(),
      [&](int lhs, int rhs) {
        return a.scores[static_cast<std::size_t>(lhs)] >
               a.scores[static_cast<std::size_t>(rhs)];
      });

  for (const int index : order) {
    std::vector<int> mask;
    for (const double value :
         a.masks[static_cast<std::size_t>(index)]) {
      mask.push_back(value >= a.threshold ? 1 : 0);
    }
    if (consistent(a, mask)) {
      return {index, mask};
    }
  }
  throw std::runtime_error("no prompt-consistent mask");
}

void print_json(const std::string& path) {
  const auto selected = select(load(path));
  std::cout << "{\"selected\":" << selected.first << ",\"mask\":[";
  for (std::size_t i = 0; i < selected.second.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << selected.second[i];
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
      const auto selected = select(load(argv[2]));
      return selected.first >= 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
