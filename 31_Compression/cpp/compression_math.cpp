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
  double temperature;
  std::vector<double> teacher_logits;
  std::vector<double> student_logits;
  double prune_threshold;
  std::vector<double> weights;
  std::vector<double> quant_values;
};

std::vector<double> parse(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    values.push_back(std::stod(token));
  }
  return values;
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
      std::stod(values.at("temperature")),
      parse(values.at("teacher_logits")),
      parse(values.at("student_logits")),
      std::stod(values.at("prune_threshold")),
      parse(values.at("weights")),
      parse(values.at("quant_values"))};
}

std::vector<double> softmax(const std::vector<double>& values) {
  double maximum = values.front();
  for (const double value : values) {
    if (value > maximum) maximum = value;
  }

  std::vector<double> output;
  double total = 0.0;
  for (const double value : values) {
    const double exponent = std::exp(value - maximum);
    output.push_back(exponent);
    total += exponent;
  }
  for (double& value : output) value /= total;
  return output;
}

double distillation_loss(const Artifact& a) {
  std::vector<double> teacher_scaled;
  std::vector<double> student_scaled;
  for (const double value : a.teacher_logits) {
    teacher_scaled.push_back(value / a.temperature);
  }
  for (const double value : a.student_logits) {
    student_scaled.push_back(value / a.temperature);
  }

  const auto teacher = softmax(teacher_scaled);
  const auto student = softmax(student_scaled);
  double cross_entropy = 0.0;
  for (std::size_t i = 0; i < teacher.size(); ++i) {
    cross_entropy -= teacher[i] * std::log(student[i]);
  }
  return a.temperature * a.temperature * cross_entropy;
}

std::vector<double> prune(const Artifact& a) {
  std::vector<double> output;
  for (const double value : a.weights) {
    output.push_back(
        std::abs(value) < a.prune_threshold ? 0.0 : value);
  }
  return output;
}

double sparsity(const std::vector<double>& values) {
  std::size_t zeros = 0;
  for (const double value : values) {
    if (value == 0.0) ++zeros;
  }
  return static_cast<double>(zeros) / values.size();
}

struct Quantized {
  double scale;
  std::vector<int> values;
  std::vector<double> dequantized;
  double max_error;
};

Quantized quantize(const std::vector<double>& values) {
  double maximum = 0.0;
  for (const double value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  const double scale = maximum == 0.0 ? 1.0 : maximum / 127.0;

  std::vector<int> quantized;
  std::vector<double> dequantized;
  double max_error = 0.0;

  for (const double value : values) {
    const long rounded = std::lround(value / scale);
    const int clipped = static_cast<int>(
        std::max(-127L, std::min(127L, rounded)));
    quantized.push_back(clipped);

    const double restored = scale * clipped;
    dequantized.push_back(restored);
    max_error = std::max(
        max_error,
        std::abs(value - restored));
  }

  return {scale, quantized, dequantized, max_error};
}

void print_double_array(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_int_array(const std::vector<int>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_json(const std::string& path) {
  const Artifact a = load(path);
  const auto pruned = prune(a);
  const auto quantized = quantize(a.quant_values);

  std::cout << std::setprecision(17)
            << "{\"distillation_loss\":"
            << distillation_loss(a)
            << ",\"pruned\":";
  print_double_array(pruned);
  std::cout << ",\"sparsity\":" << sparsity(pruned)
            << ",\"scale\":" << quantized.scale
            << ",\"quantized\":";
  print_int_array(quantized.values);
  std::cout << ",\"dequantized\":";
  print_double_array(quantized.dequantized);
  std::cout << ",\"max_error\":" << quantized.max_error
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
      const auto quantized = quantize(load(argv[2]).quant_values);
      return quantized.values.size() == 5U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
