cmake_minimum_required(VERSION 3.20)

project(cv9x_ml_computer_vision LANGUAGES CXX)

option(CV9X_ENABLE_CUDA "Enable CUDA language and GPU-native examples when toolkit is available" OFF)
option(CV9X_WARNINGS_AS_ERRORS "Treat C++ warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(cv9x_add_native_example target source)
  add_executable(${target} ${source})

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(CV9X_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

if(CV9X_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CV9X_ENABLE_CUDA=ON but no CUDA compiler was found")
  endif()

  enable_language(CUDA)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

enable_testing()

cv9x_add_native_example(
  cv9x_native_sanity
  cpp/00_Setup/native_sanity.cpp
)

add_test(
  NAME cv9x_native_sanity_self_test
  COMMAND cv9x_native_sanity --self-test
)
