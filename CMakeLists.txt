cmake_minimum_required(VERSION 3.20)

project(cv9x_ml_computer_vision VERSION 0.1.0 LANGUAGES CXX)

option(CV9X_ENABLE_CUDA "Enable CUDA language and GPU-native examples when toolkit is available" OFF)
option(CV9X_WARNINGS_AS_ERRORS "Treat C++ warnings as errors" OFF)
option(CV9X_ENABLE_CLANG_TIDY "Run clang-tidy during native compilation" OFF)

if(CV9X_ENABLE_CLANG_TIDY)
  find_program(CV9X_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
  set(CMAKE_CXX_CLANG_TIDY
      "${CV9X_CLANG_TIDY_EXE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
endif()

if(CV9X_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CV9X_ENABLE_CUDA=ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Root CMake is orchestration only. Module-specific native code lives in the module.
add_subdirectory(00_Setup)
add_subdirectory(01_TensorsMemory)
add_subdirectory(02_ProbabilityStatistics)
add_subdirectory(03_Optimization)
add_subdirectory(04_Autograd)
add_subdirectory(05_DataDiscipline)
add_subdirectory(06_LinearClassifiers)
