#include <cmath>
#include <cuda_runtime.h>

struct Candidate {
  float cx;
  float cy;
  float width;
  float height;
  float score;
};

__device__ float sigmoid_device(float value) {
  return 1.0F / (1.0F + expf(-value));
}

__global__ void decode_candidates(
    const float* predictions,
    Candidate* output,
    int count,
    int row_width,
    int grid_width,
    float anchor_width,
    float anchor_height,
    float image_width,
    float image_height) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const float* row = predictions + index * row_width;
  const int gx = index % grid_width;
  const int gy = index / grid_width;

  Candidate candidate{};
  candidate.cx = (sigmoid_device(row[0]) + gx) / grid_width;
  candidate.cy = (sigmoid_device(row[1]) + gy) / (count / grid_width);
  candidate.width = anchor_width * expf(row[2]) / image_width;
  candidate.height = anchor_height * expf(row[3]) / image_height;
  candidate.score = sigmoid_device(row[4]);
  output[index] = candidate;
}

int main() {
  // This target is an optional compile-time CUDA checkpoint.
  // Full device allocation, launch timing, class scoring and GPU NMS are
  // exercises so CPU-only CI remains portable.
  return 0;
}
