/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2022 Bogdan Simion
 * -------------
 */

#include "kernels.h"
#include <cstdint>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

__constant__ int8_t d_filter_constant[FILTER_LENGTH_MAX];

GpuTimes run_kernel6(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height)
{
    GpuTimer timer = GpuTimer(6);
    timer.start();
    
    // int32_t *d_input;
    int32_t *d_output;
    cudaTextureObject_t texInput;
    int32_t filter_size = dimension * dimension * sizeof(int8_t);
    int32_t image_size_bytes = width * height * sizeof(int32_t);

   // Calculate grid and block sizes
   dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
   dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    int32_t total_blocks = dimGrid.x * dimGrid.y;

    // min/max variables
    int32_t* d_min_array = nullptr;
    int32_t* d_max_array = nullptr;
    int32_t* h_min_array = nullptr;;
    int32_t* h_max_array = nullptr;;
    cudaHostAlloc((void**)&h_min_array, total_blocks * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_max_array, total_blocks * sizeof(int32_t), cudaHostAllocDefault);

    // Allocate device memory
    timer.start_allocate();
    //cudaMalloc((void**)&d_input, image_size_bytes);
    cudaMalloc((void**)&d_output, image_size_bytes);
    cudaMalloc(&d_min_array, total_blocks * sizeof(int32_t));
    cudaMalloc(&d_max_array, total_blocks * sizeof(int32_t));
    timer.stop_allocate();

  size_t width_in_bytes = width * sizeof(int32_t);
size_t height_in_bytes = height * sizeof(int32_t);
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int32_t>();

// Allocate CUDA array
cudaArray* cuArray;
cudaMallocArray(&cuArray, &channelDesc, width, height);


    // Copy data to device
    timer.start_tx_in();
    cudaMemcpyToSymbol(d_filter_constant, filter, filter_size);
    //cudaMemcpy(d_input, input, image_size_bytes, cudaMemcpyHostToDevice);

    // Copy data from host input to the cudaArray
    cudaMemcpy2DToArray(cuArray,   // Destination array
                        0, 0,      // Offset in the array (top left corner)
                        input,     // Source array (host memory)
                        width_in_bytes, // Source line pitch (width in bytes)
                        width_in_bytes, // Width of the matrix transfer (bytes)
                        height,    // Number of rows to copy
                        cudaMemcpyHostToDevice); // Kind of transfer

    timer.stop_tx_in();

    // Specify texture object parameters
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // Create texture object
    cudaCreateTextureObject(&texInput, &resDesc, &texDesc, NULL);


    timer.start_compute();
    kernel6<<<dimGrid, dimBlock>>>(dimension, texInput, d_output, d_min_array, d_max_array, width, height);

    int32_t h_min, h_max;
    reduce_min_max(h_min_array, d_min_array, h_min, h_max_array, d_max_array, h_max, total_blocks);

    normalize6<<<dimGrid, dimBlock>>>(d_output, width, height, h_min, h_max);
    timer.stop_compute();

    transfer_to_host(output, timer, d_output, image_size_bytes);

    // Free device memory
    timer.start_free();
    //cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_min_array);
    cudaFree(d_max_array);
    cudaFreeHost(h_min_array);
    cudaFreeHost(h_max_array);
    timer.stop_free();

    timer.stop();

    // timer.print_times();
    return timer.get_times();
}


__global__ void kernel6(int32_t dimension, cudaTextureObject_t texInput, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height)
{
    __shared__ int min_cache[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];
    __shared__ int max_cache[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    int temp_min = INT_MAX;
    int temp_max = INT_MIN;


   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   if (col < width && row < height)
   {
      int sum = 0;
      for (int i = 0; i < dimension; i++)
      {
         for (int j = 0; j < dimension; j++)
         {
            int x = col + j - dimension / 2;
            int y = row + i - dimension / 2;
            int pixel = 0;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
               pixel = tex2D<int32_t>(texInput, x, y);
            }
            sum += d_filter_constant[i * dimension + j] * pixel;
         }
      }
      output[row * width + col] = sum;
        temp_min = min(temp_min, sum);
        temp_max = max(temp_max, sum);
   }

    min_cache[threadIdx.x][threadIdx.y] = temp_min;
    max_cache[threadIdx.x][threadIdx.y] = temp_max;
    __syncthreads();

    // First Stage: Reduce along x-direction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            min_cache[threadIdx.y][threadIdx.x] = min(min_cache[threadIdx.y][threadIdx.x], min_cache[threadIdx.y][threadIdx.x + stride]);
            max_cache[threadIdx.y][threadIdx.x] = max(max_cache[threadIdx.y][threadIdx.x], max_cache[threadIdx.y][threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Second Stage: Reduce along y-direction
    if (threadIdx.x == 0) { // Only the first column participates
        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                min_cache[threadIdx.y][0] = min(min_cache[threadIdx.y][0], min_cache[threadIdx.y + stride][0]);
                max_cache[threadIdx.y][0] = max(max_cache[threadIdx.y][0], max_cache[threadIdx.y + stride][0]);
            }
            __syncthreads();
        }
    }
    // Store block-level results
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_min[blockIdx.x + blockIdx.y * gridDim.x] = min_cache[0][0];
        d_max[blockIdx.x + blockIdx.y * gridDim.x] = max_cache[0][0];
    } 
}

__global__ void normalize6(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest)
{
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   if (col < width && row < height)
   {
      int index = row * width + col;
        if (biggest != smallest)
        {
            // Perform floating-point arithmetic to avoid truncation issues
            float normalized = 255.0f * (image[index] - smallest) / (biggest - smallest);
            image[index] = static_cast<int32_t>(normalized);
        }
        // Clamp the value to the range [0, 255]
        image[index] = max(0, min(255, image[index]));
   }
}
