#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Artifact {
  int image_h;
  int image_w;
  int patch;
  std::vector<int> masked_indices;
  std::vector<double> decoder_weight;
  std::vector<double> decoder_bias;
  std::vector<double> image;
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

  return {
      std::stoi(values.at("image_h")),
      std::stoi(values.at("image_w")),
      std::stoi(values.at("patch")),
      parse_ints(values.at("masked_indices")),
      parse_doubles(values.at("decoder_weight")),
      parse_doubles(values.at("decoder_bias")),
      parse_doubles(values.at("image"))};
}

std::vector<std::vector<double>> patchify(const Artifact& a) {
  std::vector<std::vector<double>> patches;
  for (int top = 0; top < a.image_h; top += a.patch) {
    for (int left = 0; left < a.image_w; left += a.patch) {
      std::vector<double> patch;
      for (int y = 0; y < a.patch; ++y) {
        for (int x = 0; x < a.patch; ++x) {
          patch.push_back(
              a.image[static_cast<std::size_t>(
                  (top + y) * a.image_w + left + x)]);
        }
      }
      patches.push_back(patch);
    }
  }
  return patches;
}

std::vector<std::vector<double>> reconstruct(const Artifact& a) {
  const auto patches = patchify(a);
  const std::set<int> masked(
      a.masked_indices.begin(),
      a.masked_indices.end());

  const int dimensions = a.patch * a.patch;
  std::vector<double> mean(
      static_cast<std::size_t>(dimensions),
      0.0);
  int visible_count = 0;

  for (std::size_t index = 0; index < patches.size(); ++index) {
    if (masked.count(static_cast<int>(index)) != 0) continue;
    ++visible_count;
    for (int dim = 0; dim < dimensions; ++dim) {
      mean[static_cast<std::size_t>(dim)] +=
          patches[index][static_cast<std::size_t>(dim)];
    }
  }
  for (double& value : mean) value /= visible_count;

  std::vector<double> prediction(
      static_cast<std::size_t>(dimensions),
      0.0);
  for (int out = 0; out < dimensions; ++out) {
    double value = a.decoder_bias[static_cast<std::size_t>(out)];
    for (int in = 0; in < dimensions; ++in) {
      value += a.decoder_weight[
                   static_cast<std::size_t>(out * dimensions + in)]
               * mean[static_cast<std::size_t>(in)];
    }
    prediction[static_cast<std::size_t>(out)] = value;
  }

  return std::vector<std::vector<double>>(
      a.masked_indices.size(),
      prediction);
}

double loss(const Artifact& a) {
  const auto patches = patchify(a);
  const auto prediction = reconstruct(a);
  double total = 0.0;
  std::size_t count = 0;

  for (std::size_t row = 0; row < a.masked_indices.size(); ++row) {
    const auto& target =
        patches[static_cast<std::size_t>(a.masked_indices[row])];
    for (std::size_t dim = 0; dim < target.size(); ++dim) {
      const double difference = prediction[row][dim] - target[dim];
      total += difference * difference;
      ++count;
    }
  }
  return total / static_cast<double>(count);
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto prediction = reconstruct(a);

  std::cout << std::setprecision(17) << "{\"reconstruction\":[";
  for (std::size_t row = 0; row < prediction.size(); ++row) {
    if (row != 0U) std::cout << ",";
    std::cout << "[";
    for (std::size_t col = 0; col < prediction[row].size(); ++col) {
      if (col != 0U) std::cout << ",";
      std::cout << prediction[row][col];
    }
    std::cout << "]";
  }
  std::cout << "],\"loss\":" << loss(a) << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return reconstruct(load(argv[2])).size() == 2U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
