#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Instance {
  int class_id;
  int instance_id;
  double score;
  std::vector<double> mask;
};

struct Artifact {
  int height;
  int width;
  std::vector<int> semantic;
  std::vector<Instance> instances;
  double threshold;
};

std::vector<double> parse_doubles(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stod(token));
  return values;
}

std::vector<int> parse_ints(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) values.push_back(std::stoi(token));
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

  const int height = std::stoi(values.at("height"));
  const int width = std::stoi(values.at("width"));
  const auto semantic = parse_ints(values.at("semantic"));
  const auto classes = parse_ints(values.at("instance_classes"));
  const auto ids = parse_ints(values.at("instance_ids"));
  const auto scores = parse_doubles(values.at("scores"));

  std::vector<std::vector<double>> masks;
  std::stringstream rows(values.at("masks"));
  std::string token;
  while (std::getline(rows, token, ';')) masks.push_back(parse_doubles(token));

  if (classes.size() != ids.size() ||
      classes.size() != scores.size() ||
      classes.size() != masks.size()) {
    throw std::runtime_error("instance array mismatch");
  }

  std::vector<Instance> instances;
  for (std::size_t i = 0; i < classes.size(); ++i) {
    instances.push_back({classes[i], ids[i], scores[i], masks[i]});
  }

  return {
      height,
      width,
      semantic,
      instances,
      std::stod(values.at("threshold"))};
}

int encode(int class_id, int instance_id) {
  return class_id * 1000 + instance_id;
}

std::vector<int> compose(const Artifact& a) {
  std::vector<int> output;
  output.reserve(a.semantic.size());
  for (const int class_id : a.semantic) {
    output.push_back(encode(class_id, 0));
  }

  std::vector<bool> claimed(output.size(), false);
  auto instances = a.instances;
  std::stable_sort(
      instances.begin(),
      instances.end(),
      [](const Instance& lhs, const Instance& rhs) {
        return lhs.score > rhs.score;
      });

  for (const auto& instance : instances) {
    const int segment_id = encode(instance.class_id, instance.instance_id);
    for (std::size_t pixel = 0; pixel < output.size(); ++pixel) {
      if (instance.mask[pixel] >= a.threshold && !claimed[pixel]) {
        output[pixel] = segment_id;
        claimed[pixel] = true;
      }
    }
  }
  return output;
}

void print_json(const std::string& path) {
  const auto output = compose(load(path));
  std::cout << "{\"panoptic\":[";
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (i != 0U) std::cout << ",";
    std::cout << output[i];
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
      const Artifact a = load(argv[2]);
      return compose(a).size() ==
                     static_cast<std::size_t>(a.height * a.width)
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
