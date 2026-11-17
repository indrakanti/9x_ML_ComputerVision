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

struct Artifact {
  int channels{0};
  int height{0};
  int width{0};
  std::vector<double> weights;
  std::vector<double> bias;
};

std::vector<double> parse_vector(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

Artifact load_artifact(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open artifact");

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    const auto sep = line.find('=');
    if (sep == std::string::npos) throw std::runtime_error("invalid artifact line");
    values[line.substr(0, sep)] = line.substr(sep + 1);
  }

  if (std::stoi(values.at("version")) != 1) {
    throw std::runtime_error("unsupported artifact version");
  }

  Artifact artifact;
  artifact.channels = std::stoi(values.at("channels"));
  artifact.height = std::stoi(values.at("height"));
  artifact.width = std::stoi(values.at("width"));
  artifact.weights = parse_vector(values.at("weights"));
  artifact.bias = parse_vector(values.at("bias"));
  return artifact;
}

std::vector<double> example_input() {
  return {1.0, -1.0, 0.5, 2.0, 0.25, 1.5, -0.5, 1.0};
}

std::vector<double> forward(const Artifact& artifact,
                            const std::vector<double>& input) {
  const int spatial = artifact.height * artifact.width;
  if (input.size() != static_cast<std::size_t>(artifact.channels * spatial)) {
    throw std::runtime_error("input size mismatch");
  }

  std::vector<double> output(input.size(), 0.0);
  for (int y = 0; y < artifact.height; ++y) {
    for (int x = 0; x < artifact.width; ++x) {
      const int pixel = y * artifact.width + x;
      for (int out = 0; out < artifact.channels; ++out) {
        double value = artifact.bias[static_cast<std::size_t>(out)];
        for (int in = 0; in < artifact.channels; ++in) {
          const auto weight_index = static_cast<std::size_t>(
              out * artifact.channels + in);
          const auto input_index = static_cast<std::size_t>(
              in * spatial + pixel);
          value += artifact.weights[weight_index] * input[input_index];
        }
        const auto skip_index = static_cast<std::size_t>(
            out * spatial + pixel);
        output[skip_index] = std::max(0.0, value + input[skip_index]);
      }
    }
  }
  return output;
}

void print_json(const std::string& path) {
  const auto output = forward(load_artifact(path), example_input());
  std::cout << std::setprecision(17) << "{\"values\":[";
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << output[i];
  }
  std::cout << "]}\n";
}

int self_test(const std::string& path) {
  return forward(load_artifact(path), example_input()).size() == 8U
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return self_test(argv[2]);
    }
    std::cout << "Usage: cv9x_residual_block --json|--self-test <artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
