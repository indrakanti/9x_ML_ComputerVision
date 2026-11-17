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
  int version{0};
  int input_h{0};
  int input_w{0};
  int conv_out{0};
  int kernel{0};
  int padding{0};
  std::vector<double> conv_weights;
  std::vector<double> conv_bias;
  std::vector<double> fc_weights;
  std::vector<double> fc_bias;
};

std::vector<double> parse_vector(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    values.push_back(std::stod(token));
  }
  return values;
}

Artifact load_artifact(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open artifact");
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid artifact line");
    }
    values[line.substr(0, separator)] = line.substr(separator + 1);
  }

  Artifact artifact;
  artifact.version = std::stoi(values.at("version"));
  artifact.input_h = std::stoi(values.at("input_h"));
  artifact.input_w = std::stoi(values.at("input_w"));
  artifact.conv_out = std::stoi(values.at("conv_out"));
  artifact.kernel = std::stoi(values.at("kernel"));
  artifact.padding = std::stoi(values.at("padding"));
  artifact.conv_weights = parse_vector(values.at("conv_weights"));
  artifact.conv_bias = parse_vector(values.at("conv_bias"));
  artifact.fc_weights = parse_vector(values.at("fc_weights"));
  artifact.fc_bias = parse_vector(values.at("fc_bias"));

  if (artifact.version != 1) {
    throw std::runtime_error("unsupported artifact version");
  }
  if (artifact.conv_weights.size() != static_cast<std::size_t>(
          artifact.conv_out * artifact.kernel * artifact.kernel)) {
    throw std::runtime_error("conv weight count mismatch");
  }
  if (artifact.conv_bias.size() != static_cast<std::size_t>(artifact.conv_out)) {
    throw std::runtime_error("conv bias count mismatch");
  }
  if (artifact.fc_weights.size() != static_cast<std::size_t>(2 * artifact.conv_out) ||
      artifact.fc_bias.size() != 2U) {
    throw std::runtime_error("classifier shape mismatch");
  }
  return artifact;
}

std::vector<double> vertical_example() {
  return {
      0.0, 1.0, 1.0, 0.0,
      0.0, 1.0, 1.0, 0.0,
      0.0, 1.0, 1.0, 0.0,
      0.0, 1.0, 1.0, 0.0,
  };
}

std::vector<double> infer(const Artifact& artifact,
                          const std::vector<double>& image) {
  if (image.size() != static_cast<std::size_t>(
          artifact.input_h * artifact.input_w)) {
    throw std::invalid_argument("image size mismatch");
  }

  std::vector<double> pooled(static_cast<std::size_t>(artifact.conv_out), 0.0);
  for (int output = 0; output < artifact.conv_out; ++output) {
    double total = 0.0;
    for (int y = 0; y < artifact.input_h; ++y) {
      for (int x = 0; x < artifact.input_w; ++x) {
        double value = artifact.conv_bias[static_cast<std::size_t>(output)];
        for (int ky = 0; ky < artifact.kernel; ++ky) {
          for (int kx = 0; kx < artifact.kernel; ++kx) {
            const int iy = y + ky - artifact.padding;
            const int ix = x + kx - artifact.padding;
            if (iy >= 0 && iy < artifact.input_h &&
                ix >= 0 && ix < artifact.input_w) {
              const int weight_index =
                  output * artifact.kernel * artifact.kernel +
                  ky * artifact.kernel + kx;
              const int image_index = iy * artifact.input_w + ix;
              value += image[static_cast<std::size_t>(image_index)] *
                       artifact.conv_weights[
                           static_cast<std::size_t>(weight_index)];
            }
          }
        }
        total += std::max(0.0, value);
      }
    }
    pooled[static_cast<std::size_t>(output)] =
        total / static_cast<double>(artifact.input_h * artifact.input_w);
  }

  std::vector<double> logits(2, 0.0);
  for (int output = 0; output < 2; ++output) {
    double value = artifact.fc_bias[static_cast<std::size_t>(output)];
    for (int channel = 0; channel < artifact.conv_out; ++channel) {
      value += pooled[static_cast<std::size_t>(channel)] *
               artifact.fc_weights[
                   static_cast<std::size_t>(output * artifact.conv_out + channel)];
    }
    logits[static_cast<std::size_t>(output)] = value;
  }
  return logits;
}

int self_test(const std::string& artifact_path) {
  const Artifact artifact = load_artifact(artifact_path);
  const auto logits = infer(artifact, vertical_example());
  if (logits.size() != 2U) {
    return EXIT_FAILURE;
  }
  for (const double value : logits) {
    if (!std::isfinite(value)) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

void print_json(const std::string& artifact_path) {
  const Artifact artifact = load_artifact(artifact_path);
  const auto logits = infer(artifact, vertical_example());
  std::cout << std::setprecision(17)
            << "{\"logits\":["
            << logits[0] << "," << logits[1] << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return self_test(argv[2]);
    }
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    std::cout << "Usage: cv9x_tiny_cnn --self-test|--json <artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
