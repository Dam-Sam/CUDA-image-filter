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

GpuTimes run_kernel2(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height)
{
    GpuTimer timer = GpuTimer(2);
    timer.start();

    int32_t *d_input, *d_output;
    int8_t* d_filter;
    int32_t filter_size = dimension * dimension * sizeof(int8_t);
    int32_t image_size_bytes = width * height * sizeof(int32_t);

    // Launching the kernel with one thread per pixel in row-major order
    int32_t threads_per_block = THREADS_PER_BLOCK;
    int32_t total_blocks = (width * height + threads_per_block - 1) / threads_per_block;

    int32_t* d_min_array = nullptr;
    int32_t* d_max_array = nullptr;
    int32_t* h_min_array = new int32_t[total_blocks];
    int32_t* h_max_array = new int32_t[total_blocks];

    allocate_device_memory(d_input, d_output, image_size_bytes, d_filter, filter_size, d_min_array, d_max_array,
                           total_blocks);

    transfer_to_device(input, d_input, image_size_bytes, filter, d_filter, filter_size, timer);

    timer.start_compute();
    kernel2<<<total_blocks, threads_per_block>>>(d_filter, dimension, d_input, d_output, d_min_array, d_max_array,
                                                 width, height);

    int32_t h_min, h_max;
    reduce_min_max(h_min_array, d_min_array, h_min, h_max_array, d_max_array, h_max, total_blocks);

    normalize2<<<total_blocks, threads_per_block>>>(d_output, width, height, h_min, h_max);
    timer.stop_compute();

    transfer_to_host(output, timer, d_output, image_size_bytes);

    free_memory(d_input, d_output, d_filter, d_min_array, d_max_array, h_min_array, h_max_array);

    timer.stop();

    return timer.get_times();
}

__global__ void kernel2(const int8_t* filter, int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    __shared__ int min_cache[THREADS_PER_BLOCK];
    __shared__ int max_cache[THREADS_PER_BLOCK];

    int temp_min = INT_MAX;
    int temp_max = INT_MIN;

    // Each thread processes one pixel in row-major order
    while (thread_id < width * height)
    {
        int row = thread_id / width;
        int col = thread_id % width;

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
                    pixel = input[y * width + x];
                }
                sum += filter[i * dimension + j] * pixel;
            }
        }
        output[row * width + col] = sum;
        temp_min = min(temp_min, sum);
        temp_max = max(temp_max, sum);
        thread_id += total_threads;
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

    if (threadIdx.x == 0)
    {
        d_min[blockIdx.x] = min_cache[0];
        d_max[blockIdx.x] = max_cache[0];
    }
}

__global__ void normalize2(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    while (thread_id < width * height)
    {
        int row = thread_id / width;
        int col = thread_id % width;

        int index = row * width + col;
        if (biggest != smallest)
        {
            float normalized = 255.0f * (image[index] - smallest) / (biggest - smallest);
            image[index] = static_cast<int32_t>(normalized);
        }
        // Make sure between 0 and 255
        image[index] = max(0, min(255, image[index]));
        thread_id += total_threads;
    }
}
