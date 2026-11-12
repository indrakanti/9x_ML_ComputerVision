#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffsetBasis64 = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime64 = 1099511628211ULL;

std::uint64_t stable_hash64(const std::string& text) {
  std::uint64_t value = kFnvOffsetBasis64;
  for (const unsigned char byte : text) {
    value ^= static_cast<std::uint64_t>(byte);
    value *= kFnvPrime64;
  }
  return value;
}

std::string assign_split(const std::string& sample_id) {
  if (sample_id.empty()) {
    throw std::invalid_argument("sample_id must be non-empty");
  }

  const std::uint64_t bucket = stable_hash64(sample_id) % 10000ULL;
  if (bucket < 7000ULL) {
    return "train";
  }
  if (bucket < 8500ULL) {
    return "validation";
  }
  return "test";
}

double safe_divide(int numerator, int denominator) {
  return denominator == 0 ? 0.0 : static_cast<double>(numerator) / denominator;
}

struct BinaryMetrics {
  int tp{0};
  int tn{0};
  int fp{0};
  int fn{0};

  int total() const { return tp + tn + fp + fn; }
  double accuracy() const { return safe_divide(tp + tn, total()); }
  double precision() const { return safe_divide(tp, tp + fp); }
  double recall() const { return safe_divide(tp, tp + fn); }
  double specificity() const { return safe_divide(tn, tn + fp); }

  double f1() const {
    const double p = precision();
    const double r = recall();
    return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
  }
};

BinaryMetrics binary_metrics(const std::vector<int>& truth,
                             const std::vector<int>& prediction) {
  if (truth.empty() || truth.size() != prediction.size()) {
    throw std::invalid_argument("truth and prediction must be non-empty and equal length");
  }

  BinaryMetrics result;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const int expected = truth[i];
    const int actual = prediction[i];
    if ((expected != 0 && expected != 1) || (actual != 0 && actual != 1)) {
      throw std::invalid_argument("binary labels must be 0 or 1");
    }

    if (expected == 1 && actual == 1) {
      ++result.tp;
    } else if (expected == 0 && actual == 0) {
      ++result.tn;
    } else if (expected == 0 && actual == 1) {
      ++result.fp;
    } else {
      ++result.fn;
    }
  }
  return result;
}

std::map<std::string, std::string> example_splits() {
  const std::vector<std::string> sample_ids{
      "frame_0001", "frame_0002", "frame_0003", "frame_0004", "frame_0005"};
  std::map<std::string, std::string> splits;
  for (const auto& sample_id : sample_ids) {
    splits.emplace(sample_id, assign_split(sample_id));
  }
  return splits;
}

BinaryMetrics example_metrics() {
  return binary_metrics({1, 1, 1, 0, 0, 0}, {1, 1, 0, 1, 0, 0});
}

int self_test() {
  const auto metrics = example_metrics();
  if (metrics.tp != 2 || metrics.tn != 2 || metrics.fp != 1 || metrics.fn != 1) {
    std::cerr << "confusion matrix self-test failed\n";
    return EXIT_FAILURE;
  }
  if (metrics.accuracy() < 0.66 || metrics.accuracy() > 0.67) {
    std::cerr << "metric self-test failed\n";
    return EXIT_FAILURE;
  }
  if (assign_split("frame_0001") != assign_split("frame_0001")) {
    std::cerr << "split determinism self-test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void print_json() {
  const auto splits = example_splits();
  const auto metrics = example_metrics();

  std::cout << std::setprecision(17) << "{\"splits\":{";
  bool first = true;
  for (const auto& [sample_id, split] : splits) {
    if (!first) {
      std::cout << ",";
    }
    first = false;
    std::cout << "\"" << sample_id << "\":\"" << split << "\"";
  }

  std::cout << "},\"metrics\":{"
            << "\"tp\":" << metrics.tp << ","
            << "\"tn\":" << metrics.tn << ","
            << "\"fp\":" << metrics.fp << ","
            << "\"fn\":" << metrics.fn << ","
            << "\"accuracy\":" << metrics.accuracy() << ","
            << "\"precision\":" << metrics.precision() << ","
            << "\"recall\":" << metrics.recall() << ","
            << "\"specificity\":" << metrics.specificity() << ","
            << "\"f1\":" << metrics.f1()
            << "}}\n";
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

  std::cout << "Usage: cv9x_data_discipline --self-test|--json\n";
  return EXIT_SUCCESS;
}
