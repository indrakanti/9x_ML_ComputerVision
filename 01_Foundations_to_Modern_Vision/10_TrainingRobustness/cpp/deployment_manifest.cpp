#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

struct Manifest {
  int version{0};
  std::string selected_checkpoint;
  std::string dropout_inference;
  std::string augmentation_inference;
  std::string preprocessing_revision;
  int class_count{0};
};

Manifest load_manifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open deployment manifest");
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid deployment manifest line");
    }
    values[line.substr(0, separator)] = line.substr(separator + 1);
  }

  Manifest manifest;
  manifest.version = std::stoi(values.at("version"));
  manifest.selected_checkpoint = values.at("selected_checkpoint");
  manifest.dropout_inference = values.at("dropout_inference");
  manifest.augmentation_inference = values.at("augmentation_inference");
  manifest.preprocessing_revision = values.at("preprocessing_revision");
  manifest.class_count = std::stoi(values.at("class_count"));

  if (manifest.version != 1) {
    throw std::runtime_error("unsupported manifest version");
  }
  if (manifest.selected_checkpoint != "best" &&
      manifest.selected_checkpoint != "latest") {
    throw std::runtime_error("invalid checkpoint selection");
  }
  if (manifest.dropout_inference != "disabled") {
    throw std::runtime_error("dropout must be disabled for inference");
  }
  if (manifest.augmentation_inference != "disabled") {
    throw std::runtime_error("augmentation must be disabled for inference");
  }
  if (manifest.preprocessing_revision.empty()) {
    throw std::runtime_error("preprocessing revision must be non-empty");
  }
  if (manifest.class_count <= 0) {
    throw std::runtime_error("class_count must be > 0");
  }
  return manifest;
}

int self_test(const std::string& path) {
  const Manifest manifest = load_manifest(path);
  return manifest.class_count == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
}

void print_json(const std::string& path) {
  const Manifest manifest = load_manifest(path);
  std::cout
      << "{"
      << "\"selected_checkpoint\":\"" << manifest.selected_checkpoint << "\","
      << "\"dropout_inference\":\"" << manifest.dropout_inference << "\","
      << "\"augmentation_inference\":\""
      << manifest.augmentation_inference << "\","
      << "\"preprocessing_revision\":\""
      << manifest.preprocessing_revision << "\","
      << "\"class_count\":" << manifest.class_count
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
    std::cout << "Usage: cv9x_deployment_manifest --self-test|--json <manifest>\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
