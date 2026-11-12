#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct TensorContract {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    std::string dtype;
    std::string layout;
    std::string device;
    bool owns_memory;
};

std::size_t element_count(const TensorContract& contract) {
    return std::accumulate(
        contract.shape.begin(),
        contract.shape.end(),
        static_cast<std::size_t>(1),
        std::multiplies<>());
}

bool valid_rank4_nchw_contiguous(const TensorContract& contract) {
    if (contract.shape.size() != 4U || contract.strides.size() != 4U) {
        return false;
    }

    const auto n = contract.shape[0];
    const auto c = contract.shape[1];
    const auto h = contract.shape[2];
    const auto w = contract.shape[3];

    (void)n;

    const std::vector<std::size_t> expected = {
        c * h * w,
        h * w,
        w,
        1U,
    };

    return contract.layout == "NCHW" && contract.strides == expected;
}

bool run_self_test() {
    const TensorContract contract{
        {1U, 3U, 224U, 224U},
        {150528U, 50176U, 224U, 1U},
        "float32",
        "NCHW",
        "CPU",
        true,
    };

    bool passed = true;

    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    check(element_count(contract) == 150528U, "element count");
    check(valid_rank4_nchw_contiguous(contract), "contiguous NCHW strides");
    check(contract.dtype == "float32", "dtype");
    check(contract.device == "CPU", "device");
    check(contract.owns_memory, "ownership");

    if (passed) {
        std::cout << "Tensor memory-contract self-test passed.\n";
    }

    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    const bool self_test =
        argc == 2 && std::string(argv[1]) == "--self-test";

    std::cout << "Module 01: tensor memory contract\n";
    std::cout << "shape + strides + dtype + layout + device + ownership\n";

    if (self_test) {
        return run_self_test() ? 0 : 1;
    }

    std::cout << "Run with --self-test to verify the native contract.\n";
    return 0;
}
