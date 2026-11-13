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

struct LinearModel {
  int version{0};
  std::size_t input_size{0};
  std::vector<double> weights;
  double bias{0.0};
  double threshold{0.5};
};

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<double> parse_weights(const std::string& text) {
  std::vector<double> weights;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    weights.push_back(std::stod(trim(token)));
  }
  return weights;
}

LinearModel load_model(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open model artifact");
  }

  LinearModel model;
  bool have_version = false;
  bool have_input_size = false;
  bool have_weights = false;
  bool have_bias = false;
  bool have_threshold = false;

  std::string line;
  while (std::getline(input, line)) {
    line = trim(line);
    if (line.empty() || line.front() == '#') {
      continue;
    }

    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid artifact line");
    }

    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));

    if (key == "version") {
      model.version = std::stoi(value);
      have_version = true;
    } else if (key == "input_size") {
      model.input_size = static_cast<std::size_t>(std::stoul(value));
      have_input_size = true;
    } else if (key == "weights") {
      model.weights = parse_weights(value);
      have_weights = true;
    } else if (key == "bias") {
      model.bias = std::stod(value);
      have_bias = true;
    } else if (key == "threshold") {
      model.threshold = std::stod(value);
      have_threshold = true;
    }
  }

  if (!(have_version && have_input_size && have_weights && have_bias &&
        have_threshold)) {
    throw std::runtime_error("model artifact is missing required fields");
  }
  if (model.version != 1) {
    throw std::runtime_error("unsupported artifact version");
  }
  if (model.weights.size() != model.input_size) {
    throw std::runtime_error("weight count does not match input_size");
  }
  if (model.threshold < 0.0 || model.threshold > 1.0) {
    throw std::runtime_error("threshold must be in [0, 1]");
  }

  return model;
}

double linear_logit(const LinearModel& model,
                    const std::vector<double>& features) {
  if (features.size() != model.input_size) {
    throw std::invalid_argument("feature count does not match input_size");
  }

  double value = model.bias;
  for (std::size_t i = 0; i < features.size(); ++i) {
    value += features[i] * model.weights[i];
  }
  return value;
}

double sigmoid(double logit) {
  if (logit >= 0.0) {
    const double exp_neg = std::exp(-logit);
    return 1.0 / (1.0 + exp_neg);
  }
  const double exp_pos = std::exp(logit);
  return exp_pos / (1.0 + exp_pos);
}

struct InferenceResult {
  double logit;
  double probability;
  int prediction;
};

InferenceResult infer(const LinearModel& model,
                      const std::vector<double>& features) {
  const double logit = linear_logit(model, features);
  const double probability = sigmoid(logit);
  return {logit, probability, probability >= model.threshold ? 1 : 0};
}

bool near(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

int self_test(const std::string& artifact_path) {
  const LinearModel model = load_model(artifact_path);
  const InferenceResult result = infer(model, {0.5, 0.2, 0.7, 0.1});

  if (!near(result.logit, 0.9)) {
    std::cerr << "logit self-test failed\n";
    return EXIT_FAILURE;
  }
  if (!near(result.probability, sigmoid(0.9))) {
    std::cerr << "probability self-test failed\n";
    return EXIT_FAILURE;
  }
  if (result.prediction != 1) {
    std::cerr << "prediction self-test failed\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void print_json(const std::string& artifact_path) {
  const LinearModel model = load_model(artifact_path);
  const InferenceResult result = infer(model, {0.5, 0.2, 0.7, 0.1});

  std::cout << std::setprecision(17)
            << "{"
            << "\"logit\":" << result.logit << ","
            << "\"probability\":" << result.probability << ","
            << "\"prediction\":" << result.prediction
            << "}\n";
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

    std::cout
        << "Usage: cv9x_linear_classifier --self-test|--json <model-artifact>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
