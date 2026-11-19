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
  int kernel{0};
  int outputs{0};
  std::vector<double> depthwise_weights;
  std::vector<double> depthwise_bias;
  std::vector<double> pointwise_weights;
  std::vector<double> pointwise_bias;
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
  artifact.kernel = std::stoi(values.at("kernel"));
  artifact.outputs = std::stoi(values.at("outputs"));
  artifact.depthwise_weights = parse_vector(values.at("depthwise_weights"));
  artifact.depthwise_bias = parse_vector(values.at("depthwise_bias"));
  artifact.pointwise_weights = parse_vector(values.at("pointwise_weights"));
  artifact.pointwise_bias = parse_vector(values.at("pointwise_bias"));
  return artifact;
}

std::vector<double> example_input() {
  return {
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      9, 8, 7, 6, 5, 4, 3, 2, 1,
  };
}

std::vector<double> forward(const Artifact& a,
                            const std::vector<double>& input) {
  const int spatial = a.height * a.width;
  const int out_h = a.height - a.kernel + 1;
  const int out_w = a.width - a.kernel + 1;
  const int out_spatial = out_h * out_w;

  if (input.size() != static_cast<std::size_t>(a.channels * spatial)) {
    throw std::runtime_error("input size mismatch");
  }

  std::vector<double> depth(
      static_cast<std::size_t>(a.channels * out_spatial), 0.0);
  for (int channel = 0; channel < a.channels; ++channel) {
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        double value = a.depthwise_bias[static_cast<std::size_t>(channel)];
        for (int ky = 0; ky < a.kernel; ++ky) {
          for (int kx = 0; kx < a.kernel; ++kx) {
            const auto input_index = static_cast<std::size_t>(
                channel * spatial + (y + ky) * a.width + x + kx);
            const auto weight_index = static_cast<std::size_t>(
                channel * a.kernel * a.kernel + ky * a.kernel + kx);
            value += input[input_index] * a.depthwise_weights[weight_index];
          }
        }
        depth[static_cast<std::size_t>(
            channel * out_spatial + y * out_w + x)] = value;
      }
    }
  }

  std::vector<double> output(
      static_cast<std::size_t>(a.outputs * out_spatial), 0.0);
  for (int out = 0; out < a.outputs; ++out) {
    for (int pixel = 0; pixel < out_spatial; ++pixel) {
      double value = a.pointwise_bias[static_cast<std::size_t>(out)];
      for (int in = 0; in < a.channels; ++in) {
        value += a.pointwise_weights[static_cast<std::size_t>(
                     out * a.channels + in)]
                 * depth[static_cast<std::size_t>(
                     in * out_spatial + pixel)];
      }
      output[static_cast<std::size_t>(
          out * out_spatial + pixel)] = value;
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
    std::cout
        << "Usage: cv9x_depthwise_separable --json|--self-test <artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
