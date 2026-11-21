// Module 32: actual ONNX Runtime C++17 inference (not a simulated scoring fixture).
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: cv9x_module32_onnx <model.onnx> <image|text> <input.txt> <rows>\n";
            return 2;
        }
        const std::string kind = argv[2];
        const int rows = std::stoi(argv[4]);
        if (rows <= 0 || (kind != "image" && kind != "text"))
            throw std::invalid_argument("invalid rows or input kind");

        const std::size_t width = kind == "image" ? 3 * 32 * 32 : 32;
        const std::size_t count = static_cast<std::size_t>(rows) * width;
        std::ifstream input(argv[3]);
        if (!input) throw std::runtime_error("cannot open input tensor");
        std::vector<double> values;
        double value = 0.0;
        while (input >> value) {
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite input");
            values.push_back(value);
        }
        if (values.size() != count) throw std::runtime_error("input tensor size mismatch");

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "module32");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        Ort::Session session(env, argv[1], options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* inputs[] = {name.get()};
        const char* outputs[] = {output_name.get()};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape;
        Ort::Value tensor{nullptr};
        std::vector<float> float_data;
        std::vector<int64_t> int_data;
        if (kind == "image") {
            shape = {rows, 3, 32, 32};
            float_data.reserve(count);
            for (const auto item : values) float_data.push_back(static_cast<float>(item));
            tensor = Ort::Value::CreateTensor<float>(
                memory, float_data.data(), float_data.size(), shape.data(), shape.size());
        } else {
            shape = {rows, 32};
            int_data.reserve(count);
            for (const auto item : values) {
                if (std::floor(item) != item || item < 0 || item > 95)
                    throw std::runtime_error("token outside ASCII tokenizer range");
                int_data.push_back(static_cast<int64_t>(item));
            }
            tensor = Ort::Value::CreateTensor<int64_t>(
                memory, int_data.data(), int_data.size(), shape.data(), shape.size());
        }
        auto prediction = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        const auto output_shape = prediction.front().GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 2 || output_shape[0] != rows || output_shape[1] != 32)
            throw std::runtime_error("unexpected encoder output shape");
        const float* data = prediction.front().GetTensorData<float>();
        std::cout << std::fixed << std::setprecision(9);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < 32; ++j) {
                if (j) std::cout << ' ';
                std::cout << data[static_cast<std::size_t>(i) * 32 + j];
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Module 32 ONNX inference failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
