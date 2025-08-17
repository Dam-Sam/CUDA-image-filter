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

__constant__ int8_t d_filter_constant[FILTER_LENGTH_MAX];

__global__ void set_pixel5(int32_t* image, const int32_t width, const int32_t height,
                           const int32_t pixel_value)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int32_t total_pixels = width * height;


    for (int idx = thread_id; idx < total_pixels; idx += total_threads)
    {
        image[(idx / width * width + idx % width)] = pixel_value;
    }
}

GpuTimes run_kernel5(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height)
{
    GpuTimer timer = GpuTimer(5);
    timer.start();

    int32_t *d_input, *d_output;
    int32_t filter_size = dimension * dimension * sizeof(int8_t);
    int32_t image_size_bytes = width * height * sizeof(int32_t);

    int32_t threads_per_block = THREADS_PER_BLOCK;
    int32_t total_blocks = (width * height + threads_per_block - 1) / threads_per_block;

    // min/max variables
    int32_t* d_min_array = nullptr;
    int32_t* d_max_array = nullptr;
    int32_t* h_min_array = nullptr;;
    int32_t* h_max_array = nullptr;;
    cudaHostAlloc((void**)&h_min_array, total_blocks * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_max_array, total_blocks * sizeof(int32_t), cudaHostAllocDefault);

    // Allocate device memory
    timer.start_allocate();
    cudaMalloc((void**)&d_input, image_size_bytes);
    cudaMalloc((void**)&d_output, image_size_bytes);
    cudaMalloc(&d_min_array, total_blocks * sizeof(int32_t));
    cudaMalloc(&d_max_array, total_blocks * sizeof(int32_t));
    timer.stop_allocate();

    // Copy data to device
    timer.start_tx_in();
    cudaMemcpyToSymbol(d_filter_constant, filter, filter_size);
    cudaMemcpy(d_input, input, image_size_bytes, cudaMemcpyHostToDevice);
    timer.stop_tx_in();

    timer.start_compute();
    kernel5<<<total_blocks, threads_per_block>>>(dimension, d_input, d_output, d_min_array, d_max_array, width, height);

    int32_t h_min, h_max;
    reduce_min_max(h_min_array, d_min_array, h_min, h_max_array, d_max_array, h_max, total_blocks);

    if (h_min == h_max)
    {
        if (h_min < 0)
        {
            set_pixel5<<<total_blocks, threads_per_block>>>(d_output, width, height, 0);
        }
        else if (h_min > 255)
        {
            set_pixel5<<<total_blocks, threads_per_block>>>(d_output, width, height, 255);
        }
        // else - already in range no need to do anything
    }
    else
    {
        normalize5<<<total_blocks, threads_per_block>>>(d_output, width, height, h_min, h_max);
    }

    timer.stop_compute();

    transfer_to_host(output, timer, d_output, image_size_bytes);

    // Free device memory
    timer.start_free();
    cudaFree(d_input);
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

__global__ void kernel5(const int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, const int width, const int height)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    __shared__ int min_cache[THREADS_PER_BLOCK];
    __shared__ int max_cache[THREADS_PER_BLOCK];

    int temp_min = INT_MAX;
    int temp_max = INT_MIN;

    const int8_t dimension_half = dimension / 2;
    const int total_pixels = width * height;

    for (int idx = thread_id; idx < total_pixels; idx += total_threads)
    {
        const int row = idx / width;
        const int col = idx % width;
        int sum = 0;
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                int x = col + j - dimension_half;
                int y = row + i - dimension_half;
                int pixel = 0;
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    pixel = input[y * width + x];
                }
                sum += d_filter_constant[i * dimension + j] * pixel;
            }
        }
        output[row * width + col] = sum;
        temp_min = min(temp_min, sum);
        temp_max = max(temp_max, sum);
    }

    min_cache[threadIdx.x] = temp_min;
    max_cache[threadIdx.x] = temp_max;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            min_cache[threadIdx.x] = min(min_cache[threadIdx.x], min_cache[threadIdx.x + i]);
            max_cache[threadIdx.x] = max(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
        }
        __syncthreads();
    }

    // Store block-level results
    if (threadIdx.x == 0)
    {
        d_min[blockIdx.x] = min_cache[0];
        d_max[blockIdx.x] = max_cache[0];
    }
}

__global__ void normalize5(int32_t* image, const int32_t width, const int32_t height,
                           const int32_t smallest, const int32_t biggest)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int32_t range = biggest - smallest;
    const int32_t total_pixels = width * height;

    for (int idx = thread_id; idx < total_pixels; idx += total_threads)
    {
        int index = idx / width * width + idx % width;
        image[index] = 255.0f * (image[index] - smallest) / range;
    }
}
