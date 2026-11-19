cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cv9x_module10_training_robustness LANGUAGES CXX)
  option(CV9X_WARNINGS_AS_ERRORS "Treat C++ warnings as errors" OFF)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  enable_testing()
endif()

add_executable(
  cv9x_deployment_manifest
  cpp/deployment_manifest.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(
    cv9x_deployment_manifest
    PRIVATE -Wall -Wextra -Wpedantic
  )
  if(CV9X_WARNINGS_AS_ERRORS)
    target_compile_options(cv9x_deployment_manifest PRIVATE -Werror)
  endif()
endif()

add_test(
  NAME cv9x_deployment_manifest_self_test
  COMMAND cv9x_deployment_manifest
          --self-test
          ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/deployment_manifest.txt
)
