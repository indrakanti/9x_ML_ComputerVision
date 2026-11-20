cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cv9x_module19_yolo LANGUAGES CXX)
  option(CV9X_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  enable_testing()
endif()

add_executable(cv9x_yolo_decode cpp/yolo_decode.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cv9x_yolo_decode PRIVATE -Wall -Wextra -Wpedantic)
  if(CV9X_WARNINGS_AS_ERRORS)
    target_compile_options(cv9x_yolo_decode PRIVATE -Werror)
  endif()
endif()

if(CMAKE_CUDA_COMPILER)
  add_executable(cv9x_yolo_decode_cuda cpp/yolo_decode_cuda.cu)
  set_target_properties(
    cv9x_yolo_decode_cuda
    PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  )
endif()

add_test(
  NAME cv9x_yolo_decode_self_test
  COMMAND cv9x_yolo_decode --self-test
          ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/yolo_head.txt
)
