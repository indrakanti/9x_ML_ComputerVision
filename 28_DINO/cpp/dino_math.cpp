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
  double student_temperature;
  double teacher_temperature;
  std::vector<double> center;
  std::vector<double> student_a;
  std::vector<double> student_b;
  std::vector<double> teacher_a;
  std::vector<double> teacher_b;
  std::vector<double> student_params;
  std::vector<double> teacher_params;
  double teacher_momentum;
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
    const auto sep = line.find('=');
    if (sep == std::string::npos) throw std::runtime_error("invalid line");
    values[line.substr(0, sep)] = line.substr(sep + 1);
  }

  return {
      std::stoi(values.at("dimension")),
      std::stod(values.at("student_temperature")),
      std::stod(values.at("teacher_temperature")),
      parse(values.at("center")),
      parse(values.at("student_view_a")),
      parse(values.at("student_view_b")),
      parse(values.at("teacher_view_a")),
      parse(values.at("teacher_view_b")),
      parse(values.at("student_params")),
      parse(values.at("teacher_params")),
      std::stod(values.at("teacher_momentum"))};
}

std::vector<double> softmax(const std::vector<double>& input) {
  double maximum = input.front();
  for (const double value : input) {
    if (value > maximum) maximum = value;
  }

  std::vector<double> output;
  double total = 0.0;
  for (const double value : input) {
    const double exponent = std::exp(value - maximum);
    output.push_back(exponent);
    total += exponent;
  }
  for (double& value : output) value /= total;
  return output;
}

std::vector<double> student_dist(
    const std::vector<double>& logits,
    double temperature) {
  std::vector<double> scaled;
  for (const double value : logits) scaled.push_back(value / temperature);
  return softmax(scaled);
}

std::vector<double> teacher_dist(
    const std::vector<double>& logits,
    const std::vector<double>& center,
    double temperature) {
  std::vector<double> scaled;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    scaled.push_back((logits[i] - center[i]) / temperature);
  }
  return softmax(scaled);
}

double cross_entropy(
    const std::vector<double>& target,
    const std::vector<double>& prediction) {
  double value = 0.0;
  for (std::size_t i = 0; i < target.size(); ++i) {
    value -= target[i] * std::log(prediction[i]);
  }
  return value;
}

double loss(const Artifact& a) {
  const auto student_a =
      student_dist(a.student_a, a.student_temperature);
  const auto student_b =
      student_dist(a.student_b, a.student_temperature);
  const auto teacher_a =
      teacher_dist(a.teacher_a, a.center, a.teacher_temperature);
  const auto teacher_b =
      teacher_dist(a.teacher_b, a.center, a.teacher_temperature);

  return 0.5
         * (cross_entropy(teacher_b, student_a)
            + cross_entropy(teacher_a, student_b));
}

std::vector<double> ema_teacher(const Artifact& a) {
  std::vector<double> output;
  for (std::size_t i = 0; i < a.teacher_params.size(); ++i) {
    output.push_back(
        a.teacher_momentum * a.teacher_params[i]
        + (1.0 - a.teacher_momentum) * a.student_params[i]);
  }
  return output;
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

  std::cout << std::setprecision(17) << "{\"student_a\":";
  print_array(student_dist(a.student_a, a.student_temperature));
  std::cout << ",\"teacher_b\":";
  print_array(
      teacher_dist(a.teacher_b, a.center, a.teacher_temperature));
  std::cout << ",\"loss\":" << loss(a) << ",\"ema_teacher\":";
  print_array(ema_teacher(a));
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      return ema_teacher(load(argv[2])).size() == 4U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
