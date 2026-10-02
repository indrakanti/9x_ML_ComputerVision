#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

bool run_self_test() {
    bool passed = true;

    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    check(__cplusplus >= 201703L, "compiler provides C++17 or newer");

    std::vector<std::uint32_t> values = {1U, 2U, 3U, 4U};
    const std::uint32_t sum =
        std::accumulate(values.begin(), values.end(), 0U);

    check(sum == 10U, "standard-library numeric path works");
    check(sizeof(float) == 4U, "32-bit float storage is available");
    check(sizeof(std::uint8_t) == 1U, "8-bit unsigned storage is available");

    if (passed) {
        std::cout << "Native C++ runtime sanity passed.\n";
    }

    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    const bool self_test =
        argc == 2 &&
        std::string(argv[1]) == "--self-test";

    std::cout << "9x ML Computer Vision native runtime\n";
    std::cout << "C++ language level: " << __cplusplus << '\n';
    std::cout << "pointer bytes: " << sizeof(void*) << '\n';

#if defined(__GNUC__)
    std::cout << "compiler: GCC-compatible " << __VERSION__ << '\n';
#elif defined(__clang__)
    std::cout << "compiler: Clang " << __clang_version__ << '\n';
#else
    std::cout << "compiler: unknown/other\n";
#endif

    if (self_test) {
        return run_self_test() ? 0 : 1;
    }

    std::cout
        << "This target proves the native C++ toolchain. "
        << "Later lessons add LibTorch, ONNX Runtime, CUDA and TensorRT paths.\n";

    return 0;
}
