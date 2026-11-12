#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

double mean(const std::vector<double>& values) {
  if (values.empty()) {
    throw std::invalid_argument("values must be non-empty");
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double population_variance(const std::vector<double>& values) {
  const double mu = mean(values);
  double sum = 0.0;
  for (const double value : values) {
    const double delta = value - mu;
    sum += delta * delta;
  }
  return sum / static_cast<double>(values.size());
}

double covariance(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.empty() || x.size() != y.size()) {
    throw std::invalid_argument("x and y must be non-empty and the same size");
  }
  const double mx = mean(x);
  const double my = mean(y);
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sum += (x[i] - mx) * (y[i] - my);
  }
  return sum / static_cast<double>(x.size());
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
  const double sx = std::sqrt(population_variance(x));
  const double sy = std::sqrt(population_variance(y));
  if (sx == 0.0 || sy == 0.0) {
    throw std::invalid_argument("correlation undefined for zero variance");
  }
  return covariance(x, y) / (sx * sy);
}

std::pair<double, double> accuracy_confidence_interval(int correct, int total,
                                                       double z = 1.96) {
  if (total <= 0 || correct < 0 || correct > total || z <= 0.0) {
    throw std::invalid_argument("invalid confidence interval inputs");
  }
  const double p = static_cast<double>(correct) / static_cast<double>(total);
  const double se = std::sqrt(p * (1.0 - p) / static_cast<double>(total));
  const double half = z * se;
  return {std::max(0.0, p - half), std::min(1.0, p + half)};
}

bool near(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

struct Results {
  double mean_x;
  double variance_x;
  double covariance_xy;
  double correlation_xy;
  double ci_low;
  double ci_high;
};

Results compute_results() {
  const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> y{2.0, 4.0, 6.0, 8.0};
  const auto [low, high] = accuracy_confidence_interval(90, 100);
  return {mean(x), population_variance(x), covariance(x, y),
          correlation(x, y), low, high};
}

int self_test() {
  const Results r = compute_results();
  if (!near(r.mean_x, 2.5) || !near(r.variance_x, 1.25) ||
      !near(r.covariance_xy, 2.5) || !near(r.correlation_xy, 1.0)) {
    std::cerr << "statistics self-test failed\n";
    return EXIT_FAILURE;
  }
  if (!(r.ci_low < 0.90 && r.ci_high > 0.90)) {
    std::cerr << "confidence interval self-test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void print_json() {
  const Results r = compute_results();
  std::cout << std::setprecision(17)
            << "{"
            << "\"mean_x\":" << r.mean_x << ","
            << "\"variance_x\":" << r.variance_x << ","
            << "\"covariance_xy\":" << r.covariance_xy << ","
            << "\"correlation_xy\":" << r.correlation_xy << ","
            << "\"ci_low\":" << r.ci_low << ","
            << "\"ci_high\":" << r.ci_high
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    return self_test();
  }
  if (argc == 2 && std::string(argv[1]) == "--json") {
    print_json();
    return EXIT_SUCCESS;
  }
  std::cout << "Usage: cv9x_statistics --self-test|--json\n";
  return EXIT_SUCCESS;
}
