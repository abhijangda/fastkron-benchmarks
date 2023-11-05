#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
// Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
#define CUDA_LAST_ERROR do {                        \
  cudaError_t e = cudaGetLastError();               \
  if (e != cudaSuccess) {                           \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);
   CUDA_LAST_ERROR;
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   cudaFree(ptr);
   CUDA_LAST_ERROR;
}
}