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

struct Artifact {
  int dimension;
  double temperature;
  std::vector<std::vector<double>> images;
  std::vector<std::vector<double>> texts;
};

std::vector<double> parse_values(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    values.push_back(std::stod(token));
  }
  return values;
}

std::vector<std::vector<double>> parse_rows(const std::string& text) {
  std::vector<std::vector<double>> rows;
  std::stringstream stream(text);
  std::string row;
  while (std::getline(stream, row, ';')) {
    rows.push_back(parse_values(row));
  }
  return rows;
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

  return {
      std::stoi(values.at("dimension")),
      std::stod(values.at("temperature")),
      parse_rows(values.at("image_embeddings")),
      parse_rows(values.at("text_embeddings"))};
}

std::vector<double> normalize(const std::vector<double>& input) {
  double squared = 0.0;
  for (const double value : input) squared += value * value;
  const double norm = std::sqrt(squared);
  if (norm == 0.0) throw std::runtime_error("zero embedding");

  std::vector<double> output;
  output.reserve(input.size());
  for (const double value : input) output.push_back(value / norm);
  return output;
}

std::vector<std::vector<double>> similarity(const Artifact& a) {
  std::vector<std::vector<double>> images;
  std::vector<std::vector<double>> texts;

  for (const auto& row : a.images) images.push_back(normalize(row));
  for (const auto& row : a.texts) texts.push_back(normalize(row));

  std::vector<std::vector<double>> matrix;
  for (const auto& image : images) {
    std::vector<double> row;
    for (const auto& text : texts) {
      double value = 0.0;
      for (int dim = 0; dim < a.dimension; ++dim) {
        value += image[static_cast<std::size_t>(dim)]
                 * text[static_cast<std::size_t>(dim)];
      }
      row.push_back(value);
    }
    matrix.push_back(row);
  }
  return matrix;
}

std::vector<int> top_text(const std::vector<std::vector<double>>& matrix) {
  std::vector<int> output;
  for (const auto& row : matrix) {
    const auto best = std::max_element(row.begin(), row.end());
    output.push_back(static_cast<int>(std::distance(row.begin(), best)));
  }
  return output;
}

void print_json(const std::string& path) {
  const auto matrix = similarity(load(path));
  const auto matches = top_text(matrix);

  std::cout << std::setprecision(17) << "{\"similarity\":[";
  for (std::size_t r = 0; r < matrix.size(); ++r) {
    if (r != 0U) std::cout << ",";
    std::cout << "[";
    for (std::size_t c = 0; c < matrix[r].size(); ++c) {
      if (c != 0U) std::cout << ",";
      std::cout << matrix[r][c];
    }
    std::cout << "]";
  }
  std::cout << "],\"top_text\":[";
  for (std::size_t i = 0; i < matches.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << matches[i];
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
      const auto matrix = similarity(load(argv[2]));
      return matrix.size() == 3U ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
