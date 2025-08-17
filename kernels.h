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

#ifndef __KERNELS__H
#define __KERNELS__H

#include "gpu_timer.h"

// This controls threads per block for kernels 1 - 4
#define THREADS_PER_BLOCK 256

#define FILTER_LENGTH_MAX 81


/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions
 * unfortunately, so don't use those for variable names.*/

void allocate_device_memory(int32_t*& d_input, int32_t*& d_output, int32_t image_size_bytes, int8_t*& d_filter,
                            int32_t filter_size, int32_t*& d_min_array, int32_t*& d_max_array,
                            int32_t min_max_array_length);

void transfer_to_device(const int32_t* h_input, int32_t* d_input, int32_t image_size_bytes, const int8_t* h_filter,
                        int8_t* d_filter, int32_t filter_size, GpuTimer& timer);

void reduce_min_max(int32_t* h_min_array, int32_t* d_min_array, int32_t& h_min, int32_t* h_max_array,
                    int32_t* d_max_array, int32_t& h_max, int min_max_array_length);

void transfer_to_host(int32_t* h_output, GpuTimer& timer, int32_t* d_output, int32_t image_size_bytes);

void free_memory(int32_t* d_input, int32_t* d_output, int8_t* d_filter, int32_t* d_min_array, int32_t* d_max_array,
                 int32_t* h_min_array, int32_t* h_max_array);

double run_best_cpu(const int8_t* filter, int32_t dimension, const int32_t* input,
                    int32_t* output, int32_t width, int32_t height);

GpuTimes run_kernel1(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height);
__global__ void kernel1(const int8_t* filter, int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height);
__global__ void normalize1(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);

GpuTimes run_kernel2(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height);
__global__ void kernel2(const int8_t* filter, int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height);
__global__ void normalize2(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);

GpuTimes run_kernel3(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height);
__global__ void kernel3(const int8_t* filter, int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height);
__global__ void normalize3(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);

GpuTimes run_kernel4(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height);
__global__ void kernel4(const int8_t* filter, int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height);
__global__ void normalize4(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);

GpuTimes run_kernel5(const int8_t* filter, int32_t dimension, const int32_t* input,
                     int32_t* output, int32_t width, int32_t height);
/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void kernel5(int32_t dimension, const int32_t* input, int32_t* output,
                        int32_t* d_min, int32_t* d_max, int width, int height);
__global__ void normalize5(int32_t* image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);


#endif
