#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Model {
  int version{0};
  std::size_t input_size{0};
  std::size_t hidden_size{0};
  std::vector<double> mean;
  std::vector<double> stddev;
  std::vector<std::vector<double>> w1;
  std::vector<double> b1;
  std::vector<double> w2;
  double b2{0.0};
  double threshold{0.5};
};

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<double> parse_vector(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(trim(token)));
  return values;
}

std::vector<std::vector<double>> parse_matrix(const std::string& text) {
  std::vector<std::vector<double>> matrix;
  std::stringstream stream(text);
  std::string row;
  while (std::getline(stream, row, ';')) matrix.push_back(parse_vector(row));
  return matrix;
}

Model load_model(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open artifact");
  Model model;
  std::string line;
  while (std::getline(input, line)) {
    line = trim(line);
    if (line.empty() || line.front() == '#') continue;
    const auto sep = line.find('=');
    if (sep == std::string::npos) throw std::runtime_error("invalid artifact line");
    const auto key = trim(line.substr(0, sep));
    const auto value = trim(line.substr(sep + 1));
    if (key == "version") model.version = std::stoi(value);
    else if (key == "input_size") model.input_size = std::stoul(value);
    else if (key == "hidden_size") model.hidden_size = std::stoul(value);
    else if (key == "mean") model.mean = parse_vector(value);
    else if (key == "std") model.stddev = parse_vector(value);
    else if (key == "w1") model.w1 = parse_matrix(value);
    else if (key == "b1") model.b1 = parse_vector(value);
    else if (key == "w2") model.w2 = parse_vector(value);
    else if (key == "b2") model.b2 = std::stod(value);
    else if (key == "threshold") model.threshold = std::stod(value);
  }
  if (model.version != 1 || model.mean.size() != model.input_size ||
      model.stddev.size() != model.input_size || model.w1.size() != model.hidden_size ||
      model.b1.size() != model.hidden_size || model.w2.size() != model.hidden_size) {
    throw std::runtime_error("artifact shape/version validation failed");
  }
  for (const auto& row : model.w1) {
    if (row.size() != model.input_size) throw std::runtime_error("w1 shape mismatch");
  }
  return model;
}

double sigmoid(double value) {
  if (value >= 0.0) {
    const double e = std::exp(-value);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(value);
  return e / (1.0 + e);
}

double forward_logit(const Model& model, const std::vector<double>& input) {
  if (input.size() != model.input_size) throw std::invalid_argument("input size mismatch");
  std::vector<double> normalized(model.input_size);
  for (std::size_t i = 0; i < model.input_size; ++i) {
    if (model.stddev[i] <= 0.0) throw std::runtime_error("invalid std");
    normalized[i] = (input[i] - model.mean[i]) / model.stddev[i];
  }
  std::vector<double> hidden(model.hidden_size);
  for (std::size_t h = 0; h < model.hidden_size; ++h) {
    double value = model.b1[h];
    for (std::size_t i = 0; i < model.input_size; ++i) value += model.w1[h][i] * normalized[i];
    hidden[h] = std::max(0.0, value);
  }
  double logit = model.b2;
  for (std::size_t h = 0; h < model.hidden_size; ++h) logit += model.w2[h] * hidden[h];
  return logit;
}

int self_test(const std::string& path) {
  const auto model = load_model(path);
  const double logit = forward_logit(model, {0.6, 0.2, 0.8, 0.3});
  if (!std::isfinite(logit)) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

void print_json(const std::string& path) {
  const auto model = load_model(path);
  const double logit = forward_logit(model, {0.6, 0.2, 0.8, 0.3});
  const double probability = sigmoid(logit);
  const int prediction = probability >= model.threshold ? 1 : 0;
  std::cout << std::setprecision(17)
            << "{\"logit\":" << logit
            << ",\"probability\":" << probability
            << ",\"prediction\":" << prediction << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--self-test") return self_test(argv[2]);
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    std::cout << "Usage: cv9x_mlp --self-test|--json <artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
