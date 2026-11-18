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

using Matrix = std::vector<std::vector<double>>;

struct Artifact {
  int feature_dimension;
  std::vector<double> features;
  int probe_outputs;
  Matrix probe_weight;
  std::vector<double> probe_bias;
  int adapter_hidden;
  Matrix adapter_down;
  std::vector<double> adapter_down_bias;
  Matrix adapter_up;
  std::vector<double> adapter_up_bias;
  int lora_outputs;
  Matrix base_weight;
  std::vector<double> base_bias;
  int lora_rank;
  double lora_alpha;
  Matrix lora_a;
  Matrix lora_b;
};

std::vector<double> parse_vector(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

Matrix parse_matrix(const std::string& text) {
  Matrix matrix;
  std::stringstream stream(text);
  std::string row;
  while (std::getline(stream, row, ';')) matrix.push_back(parse_vector(row));
  return matrix;
}

Artifact load(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open artifact");

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid artifact line");
    }
    values[line.substr(0, separator)] = line.substr(separator + 1);
  }

  return {
      std::stoi(values.at("feature_dimension")),
      parse_vector(values.at("features")),
      std::stoi(values.at("probe_outputs")),
      parse_matrix(values.at("probe_weight")),
      parse_vector(values.at("probe_bias")),
      std::stoi(values.at("adapter_hidden")),
      parse_matrix(values.at("adapter_down")),
      parse_vector(values.at("adapter_down_bias")),
      parse_matrix(values.at("adapter_up")),
      parse_vector(values.at("adapter_up_bias")),
      std::stoi(values.at("lora_outputs")),
      parse_matrix(values.at("base_weight")),
      parse_vector(values.at("base_bias")),
      std::stoi(values.at("lora_rank")),
      std::stod(values.at("lora_alpha")),
      parse_matrix(values.at("lora_a")),
      parse_matrix(values.at("lora_b"))};
}

std::vector<double> linear(
    const std::vector<double>& vector,
    const Matrix& weight,
    const std::vector<double>& bias) {
  std::vector<double> output(weight.size(), 0.0);
  for (std::size_t out = 0; out < weight.size(); ++out) {
    double value = bias[out];
    for (std::size_t in = 0; in < vector.size(); ++in) {
      value += weight[out][in] * vector[in];
    }
    output[out] = value;
  }
  return output;
}

std::vector<double> probe(const Artifact& a) {
  return linear(a.features, a.probe_weight, a.probe_bias);
}

std::vector<double> adapter(const Artifact& a) {
  auto hidden = linear(
      a.features,
      a.adapter_down,
      a.adapter_down_bias);
  for (double& value : hidden) value = std::max(0.0, value);
  const auto delta = linear(hidden, a.adapter_up, a.adapter_up_bias);

  std::vector<double> output;
  for (std::size_t i = 0; i < a.features.size(); ++i) {
    output.push_back(a.features[i] + delta[i]);
  }
  return output;
}

std::vector<double> lora(const Artifact& a) {
  const auto base = linear(a.features, a.base_weight, a.base_bias);
  const auto low_rank = linear(
      a.features,
      a.lora_a,
      std::vector<double>(
          static_cast<std::size_t>(a.lora_rank),
          0.0));
  const auto update = linear(
      low_rank,
      a.lora_b,
      std::vector<double>(
          static_cast<std::size_t>(a.lora_outputs),
          0.0));

  const double scale = a.lora_alpha / a.lora_rank;
  std::vector<double> output;
  for (std::size_t i = 0; i < base.size(); ++i) {
    output.push_back(base[i] + scale * update[i]);
  }
  return output;
}

std::unordered_map<std::string, int> counts(const Artifact& a) {
  const int probe_count =
      a.probe_outputs * a.feature_dimension + a.probe_outputs;
  const int adapter_count =
      a.adapter_hidden * a.feature_dimension
      + a.adapter_hidden
      + a.feature_dimension * a.adapter_hidden
      + a.feature_dimension;
  const int lora_count =
      a.lora_rank * (a.feature_dimension + a.lora_outputs);
  const int full_linear =
      a.lora_outputs * a.feature_dimension + a.lora_outputs;

  return {
      {"probe", probe_count},
      {"adapter", adapter_count},
      {"lora", lora_count},
      {"full_linear", full_linear}};
}

void print_array(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto parameter_counts = counts(a);

  std::cout << std::setprecision(17) << "{\"probe\":";
  print_array(probe(a));
  std::cout << ",\"adapter\":";
  print_array(adapter(a));
  std::cout << ",\"lora\":";
  print_array(lora(a));
  std::cout << ",\"parameter_counts\":{"
            << "\"probe\":" << parameter_counts.at("probe")
            << ",\"adapter\":" << parameter_counts.at("adapter")
            << ",\"lora\":" << parameter_counts.at("lora")
            << ",\"full_linear\":" << parameter_counts.at("full_linear")
            << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return lora(load(argv[2])).size() == 2U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
