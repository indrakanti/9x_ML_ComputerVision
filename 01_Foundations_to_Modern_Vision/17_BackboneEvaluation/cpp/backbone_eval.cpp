#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Model {
  std::string name;
  double accuracy;
  double latency_ms;
  double params_m;
  double readiness;
};

struct Weights {
  double accuracy;
  double latency;
  double params;
  double readiness;
};

std::map<std::string, double> parse_pairs(const std::string& text) {
  std::map<std::string, double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto separator = token.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid key:value pair");
    }
    values[token.substr(0, separator)] =
        std::stod(token.substr(separator + 1));
  }
  return values;
}

std::pair<Weights, std::vector<Model>> load(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open artifact");
  }

  Weights weights{};
  bool have_weights = false;
  std::vector<Model> models;
  std::string line;

  while (std::getline(input, line)) {
    if (line.rfind("weights=", 0) == 0) {
      const auto values = parse_pairs(line.substr(8));
      weights = {
          values.at("accuracy"),
          values.at("latency"),
          values.at("params"),
          values.at("readiness"),
      };
      have_weights = true;
    } else if (line.rfind("backbone=", 0) == 0) {
      const auto comma = line.find(',');
      if (comma == std::string::npos) {
        throw std::runtime_error("invalid backbone row");
      }
      const std::string name = line.substr(9, comma - 9);
      const auto values = parse_pairs(line.substr(comma + 1));
      models.push_back(
          {
              name,
              values.at("accuracy"),
              values.at("latency_ms"),
              values.at("params_m"),
              values.at("readiness"),
          });
    }
  }

  if (!have_weights || models.empty()) {
    throw std::runtime_error("incomplete scorecard");
  }
  return {weights, models};
}

std::vector<std::pair<std::string, double>> rank(
    const Weights& weights,
    const std::vector<Model>& models) {
  double min_latency = models.front().latency_ms;
  double max_latency = models.front().latency_ms;
  double min_params = models.front().params_m;
  double max_params = models.front().params_m;

  for (const auto& model : models) {
    min_latency = std::min(min_latency, model.latency_ms);
    max_latency = std::max(max_latency, model.latency_ms);
    min_params = std::min(min_params, model.params_m);
    max_params = std::max(max_params, model.params_m);
  }

  const auto inverse_norm = [](double value, double low, double high) {
    return high == low ? 1.0 : 1.0 - (value - low) / (high - low);
  };

  std::vector<std::pair<std::string, double>> ranking;
  for (const auto& model : models) {
    const double score =
        weights.accuracy * model.accuracy
        + weights.latency
              * inverse_norm(
                    model.latency_ms,
                    min_latency,
                    max_latency)
        + weights.params
              * inverse_norm(
                    model.params_m,
                    min_params,
                    max_params)
        + weights.readiness * model.readiness;

    ranking.push_back({model.name, score});
  }

  std::sort(
      ranking.begin(),
      ranking.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
      });
  return ranking;
}

void print_json(const std::string& path) {
  const auto loaded = load(path);
  const auto ranking = rank(loaded.first, loaded.second);

  std::cout << std::setprecision(17) << "{\"ranking\":[";
  for (std::size_t index = 0; index < ranking.size(); ++index) {
    if (index != 0U) {
      std::cout << ",";
    }
    std::cout << "{\"name\":\"" << ranking[index].first
              << "\",\"score\":" << ranking[index].second << "}";
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--json") {
      print_json(argv[2]);
      return EXIT_SUCCESS;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      const auto loaded = load(argv[2]);
      return rank(loaded.first, loaded.second).size() == 3U
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
