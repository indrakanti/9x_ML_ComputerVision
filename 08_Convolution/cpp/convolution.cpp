#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  int stride_h{1};
  int stride_w{1};
  int pad_h{1};
  int pad_w{1};
};

struct Tensor3D {
  int channels;
  int height;
  int width;
  std::vector<double> values;

  double at(int c, int y, int x) const {
    return values[static_cast<std::size_t>((c * height + y) * width + x)];
  }
};

struct Weights4D {
  int outputs;
  int inputs;
  int kernel_h;
  int kernel_w;
  std::vector<double> values;

  double at(int o, int c, int y, int x) const {
    const int index = (((o * inputs + c) * kernel_h + y) * kernel_w + x);
    return values[static_cast<std::size_t>(index)];
  }
};

Tensor3D conv2d(const Tensor3D& image, const Weights4D& weights,
                const std::vector<double>& bias, const Config& config) {
  if (image.channels != weights.inputs) {
    throw std::invalid_argument("input-channel mismatch");
  }
  if (static_cast<int>(bias.size()) != weights.outputs) {
    throw std::invalid_argument("bias size mismatch");
  }

  const int numerator_h =
      image.height + 2 * config.pad_h - weights.kernel_h;
  const int numerator_w =
      image.width + 2 * config.pad_w - weights.kernel_w;
  if (numerator_h < 0 || numerator_w < 0 ||
      config.stride_h <= 0 || config.stride_w <= 0) {
    throw std::invalid_argument("invalid convolution geometry");
  }

  const int out_h = numerator_h / config.stride_h + 1;
  const int out_w = numerator_w / config.stride_w + 1;
  Tensor3D output{
      weights.outputs,
      out_h,
      out_w,
      std::vector<double>(
          static_cast<std::size_t>(weights.outputs * out_h * out_w), 0.0),
  };

  for (int o = 0; o < weights.outputs; ++o) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        double accumulator = bias[static_cast<std::size_t>(o)];
        for (int c = 0; c < image.channels; ++c) {
          for (int ky = 0; ky < weights.kernel_h; ++ky) {
            for (int kx = 0; kx < weights.kernel_w; ++kx) {
              const int iy = oy * config.stride_h + ky - config.pad_h;
              const int ix = ox * config.stride_w + kx - config.pad_w;
              if (iy >= 0 && iy < image.height &&
                  ix >= 0 && ix < image.width) {
                accumulator +=
                    image.at(c, iy, ix) * weights.at(o, c, ky, kx);
              }
            }
          }
        }
        output.values[static_cast<std::size_t>(
            (o * out_h + oy) * out_w + ox)] = accumulator;
      }
    }
  }
  return output;
}

Tensor3D example_image() {
  return {
      2, 3, 3,
      {
          1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
          9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
      },
  };
}

Weights4D example_weights() {
  return {
      2, 2, 2, 2,
      {
          1.0, 0.0, 0.0, -1.0,
          0.5, 0.0, 0.0, -0.5,
          0.0, 1.0, -1.0, 0.0,
          0.0, -0.25, 0.25, 0.0,
      },
  };
}

Tensor3D example_output() {
  return conv2d(
      example_image(),
      example_weights(),
      {0.25, -0.5},
      Config{},
  );
}

int self_test() {
  const Tensor3D output = example_output();
  if (output.channels != 2 || output.height != 4 || output.width != 4) {
    std::cerr << "shape self-test failed\n";
    return EXIT_FAILURE;
  }
  for (double value : output.values) {
    if (!std::isfinite(value)) {
      std::cerr << "non-finite output\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

void print_json() {
  const Tensor3D output = example_output();
  std::cout << std::setprecision(17)
            << "{\"shape\":["
            << output.channels << ","
            << output.height << ","
            << output.width << "],\"values\":[";
  for (std::size_t index = 0; index < output.values.size(); ++index) {
    if (index != 0U) {
      std::cout << ",";
    }
    std::cout << output.values[index];
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      return self_test();
    }
    if (argc == 2 && std::string(argv[1]) == "--json") {
      print_json();
      return EXIT_SUCCESS;
    }
    std::cout << "Usage: cv9x_convolution --self-test|--json\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
