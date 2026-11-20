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

struct Adapter {
  int outputs;
  std::vector<std::vector<double>> weight;
  std::vector<double> bias;
};

struct Artifact {
  std::string feature_revision;
  int feature_dimension;
  int backbone_parameters;
  std::vector<double> features;
  Adapter classification;
  Adapter geometry;
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

std::vector<std::vector<double>> parse_matrix(const std::string& text) {
  std::vector<std::vector<double>> matrix;
  std::stringstream stream(text);
  std::string row;
  while (std::getline(stream, row, ';')) {
    matrix.push_back(parse_vector(row));
  }
  return matrix;
}

Adapter parse_adapter(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& prefix) {
  return {
      std::stoi(values.at(prefix + "_outputs")),
      parse_matrix(values.at(prefix + "_weight")),
      parse_vector(values.at(prefix + "_bias"))};
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
      values.at("feature_revision"),
      std::stoi(values.at("feature_dimension")),
      std::stoi(values.at("backbone_parameters")),
      parse_vector(values.at("features")),
      parse_adapter(values, "classification"),
      parse_adapter(values, "geometry")};
}

std::vector<double> forward(
    const std::vector<double>& features,
    const Adapter& adapter) {
  std::vector<double> output(
      static_cast<std::size_t>(adapter.outputs),
      0.0);

  for (int out = 0; out < adapter.outputs; ++out) {
    double value = adapter.bias[static_cast<std::size_t>(out)];
    for (std::size_t index = 0; index < features.size(); ++index) {
      value += adapter.weight[static_cast<std::size_t>(out)][index]
               * features[index];
    }
    output[static_cast<std::size_t>(out)] = value;
  }
  return output;
}

int parameter_count(const Adapter& adapter) {
  if (adapter.weight.empty()) {
    return static_cast<int>(adapter.bias.size());
  }
  return static_cast<int>(
      adapter.weight.size() * adapter.weight.front().size()
      + adapter.bias.size());
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
  const auto classification = forward(a.features, a.classification);
  const auto geometry = forward(a.features, a.geometry);
  const int adapter_parameters =
      parameter_count(a.classification)
      + parameter_count(a.geometry);

  std::cout << std::setprecision(17)
            << "{\"feature_revision\":\""
            << a.feature_revision
            << "\",\"classification\":";
  print_array(classification);
  std::cout << ",\"geometry\":";
  print_array(geometry);
  std::cout << ",\"adapter_parameters\":"
            << adapter_parameters
            << "}\n";
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
      return a.feature_dimension == 4
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
