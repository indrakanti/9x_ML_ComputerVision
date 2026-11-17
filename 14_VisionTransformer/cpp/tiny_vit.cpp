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
  int image_h{0};
  int image_w{0};
  int patch{0};
  int embed{0};
  std::vector<double> patch_weight;
  std::vector<double> patch_bias;
  std::vector<double> position;
  std::vector<double> wq;
  std::vector<double> wk;
  std::vector<double> wv;
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

  Artifact a;
  a.image_h = std::stoi(values.at("image_h"));
  a.image_w = std::stoi(values.at("image_w"));
  a.patch = std::stoi(values.at("patch"));
  a.embed = std::stoi(values.at("embed"));
  a.patch_weight = parse_vector(values.at("patch_weight"));
  a.patch_bias = parse_vector(values.at("patch_bias"));
  a.position = parse_vector(values.at("position"));
  a.wq = parse_vector(values.at("wq"));
  a.wk = parse_vector(values.at("wk"));
  a.wv = parse_vector(values.at("wv"));
  return a;
}

std::vector<double> example_image() {
  return {
      1, 2, 3, 4,
      5, 6, 7, 8,
      8, 7, 6, 5,
      4, 3, 2, 1,
  };
}

std::vector<std::vector<double>> patchify(
    const Artifact& a,
    const std::vector<double>& image) {
  std::vector<std::vector<double>> patches;
  for (int top = 0; top < a.image_h; top += a.patch) {
    for (int left = 0; left < a.image_w; left += a.patch) {
      std::vector<double> patch;
      for (int y = 0; y < a.patch; ++y) {
        for (int x = 0; x < a.patch; ++x) {
          patch.push_back(image[static_cast<std::size_t>(
              (top + y) * a.image_w + left + x)]);
        }
      }
      patches.push_back(patch);
    }
  }
  return patches;
}

std::vector<double> linear(
    const std::vector<double>& input,
    const std::vector<double>& weights,
    const std::vector<double>& bias,
    int outputs) {
  std::vector<double> result(static_cast<std::size_t>(outputs), 0.0);
  const int inputs = static_cast<int>(input.size());
  for (int out = 0; out < outputs; ++out) {
    double value = bias[static_cast<std::size_t>(out)];
    for (int in = 0; in < inputs; ++in) {
      value += weights[static_cast<std::size_t>(out * inputs + in)]
               * input[static_cast<std::size_t>(in)];
    }
    result[static_cast<std::size_t>(out)] = value;
  }
  return result;
}

std::vector<std::vector<double>> project(
    const std::vector<std::vector<double>>& tokens,
    const std::vector<double>& weights,
    int dim) {
  std::vector<std::vector<double>> result;
  const std::vector<double> zero_bias(
      static_cast<std::size_t>(dim),
      0.0);
  for (const auto& token : tokens) {
    result.push_back(linear(token, weights, zero_bias, dim));
  }
  return result;
}

std::vector<double> forward(
    const Artifact& a,
    const std::vector<double>& image) {
  const auto patches = patchify(a, image);
  std::vector<std::vector<double>> tokens;

  for (std::size_t index = 0; index < patches.size(); ++index) {
    auto token = linear(
        patches[index],
        a.patch_weight,
        a.patch_bias,
        a.embed);
    for (int dim = 0; dim < a.embed; ++dim) {
      token[static_cast<std::size_t>(dim)] +=
          a.position[
              index * static_cast<std::size_t>(a.embed)
              + static_cast<std::size_t>(dim)];
    }
    tokens.push_back(token);
  }

  const auto q = project(tokens, a.wq, a.embed);
  const auto k = project(tokens, a.wk, a.embed);
  const auto v = project(tokens, a.wv, a.embed);
  const double scale = std::sqrt(static_cast<double>(a.embed));

  std::vector<double> output;
  for (const auto& query : q) {
    std::vector<double> scores;
    for (const auto& key : k) {
      double score = 0.0;
      for (int dim = 0; dim < a.embed; ++dim) {
        score += query[static_cast<std::size_t>(dim)]
                 * key[static_cast<std::size_t>(dim)];
      }
      scores.push_back(score / scale);
    }

    double maximum = scores.front();
    for (const double score : scores) {
      if (score > maximum) maximum = score;
    }

    std::vector<double> probabilities;
    double total = 0.0;
    for (const double score : scores) {
      const double value = std::exp(score - maximum);
      probabilities.push_back(value);
      total += value;
    }
    for (double& probability : probabilities) {
      probability /= total;
    }

    for (int dim = 0; dim < a.embed; ++dim) {
      double value = 0.0;
      for (std::size_t token = 0; token < v.size(); ++token) {
        value += probabilities[token]
                 * v[token][static_cast<std::size_t>(dim)];
      }
      output.push_back(value);
    }
  }
  return output;
}

void print_json(const std::string& path) {
  const auto output = forward(load_artifact(path), example_image());
  std::cout << std::setprecision(17) << "{\"values\":[";
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << output[i];
  }
  std::cout << "]}\n";
}

int self_test(const std::string& path) {
  return forward(load_artifact(path), example_image()).size() == 8U
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
    std::cout << "Usage: cv9x_tiny_vit --json|--self-test <artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
