#include "kernels.h"
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

void allocate_device_memory(int32_t*& d_input, int32_t*& d_output, int32_t image_size_bytes, int8_t*& d_filter,
                            int32_t filter_size, int32_t*& d_min_array, int32_t*& d_max_array,
                            int32_t min_max_array_length)
{
    cudaMalloc((void**)&d_filter, filter_size);
    cudaMalloc((void**)&d_input, image_size_bytes);
    cudaMalloc((void**)&d_output, image_size_bytes);
    cudaMalloc(&d_min_array, min_max_array_length * sizeof(int32_t));
    cudaMalloc(&d_max_array, min_max_array_length * sizeof(int32_t));
}

void transfer_to_device(const int32_t* h_input, int32_t* d_input, int32_t image_size_bytes, const int8_t* h_filter,
                        int8_t* d_filter, int32_t filter_size, GpuTimer& timer)
{
    timer.start_tx_in();
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, image_size_bytes, cudaMemcpyHostToDevice);
    timer.stop_tx_in();
}

void reduce_min_max(int32_t* h_min_array, int32_t* d_min_array, int32_t& h_min, int32_t* h_max_array,
                    int32_t* d_max_array, int32_t& h_max, int min_max_array_length)
{
    h_min = INT_MAX;
    h_max = INT_MIN;
    cudaMemcpy(h_min_array, d_min_array, min_max_array_length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_array, d_max_array, min_max_array_length * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < min_max_array_length; i++)
    {
        h_min = std::min(h_min, h_min_array[i]);
        h_max = std::max(h_max, h_max_array[i]);
    }
}

void transfer_to_host(int32_t* h_output, GpuTimer& timer, int32_t* d_output, int32_t image_size_bytes)
{
    timer.start_tx_out();
    cudaMemcpy(h_output, d_output, image_size_bytes, cudaMemcpyDeviceToHost);
    timer.stop_tx_out();
}

void free_memory(int32_t* d_input, int32_t* d_output, int8_t* d_filter, int32_t* d_min_array, int32_t* d_max_array,
                 int32_t* h_min_array, int32_t* h_max_array)
{
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_min_array);
    cudaFree(d_max_array);
    delete[] h_min_array;
    delete[] h_max_array;
}
