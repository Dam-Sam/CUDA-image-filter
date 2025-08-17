#include <iostream>
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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"
#include "kernels.h"

struct timespec start, stop;
// const int8_t FILTER[] = {
//     0,
//     1,
//     0,
//     1,
//     -4,
//     1,
//     0,
//     1,
//     0,
// };

// const int FILTER_DIMENSION = 3;

const int8_t FILTER[] = {
    0, 1, 1, 2, 2, 2, 1, 1, 0, 1, 2, 4, 5, 5, 5, 4, 2,
    1, 1, 4, 5, 3, 0, 3, 5, 4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3, -12, -24, -12,
    3, 5, 2, 1, 4, 5, 3, 0, 3, 5, 4, 1, 1, 2, 4, 5, 5,
    5, 4, 2, 1, 0, 1, 1, 2, 2, 2, 1, 1, 0,
};
const int FILTER_DIMENSION = 9;

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out)
{
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu / time_gpu_computation);
    printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
               time_gpu_transfer_out));
}

void print_times(float cpu, GpuTimes gpu)
{
    print_run(cpu, gpu.kernel, gpu.computation, gpu.tx_in, gpu.tx_out);
}

int main(int argc, char** argv)
{
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3)
    {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt(argc, argv, "i:o:")) != -1)
    {
        switch (c)
        {
        case 'i':
            input_filename = std::string(optarg);
            break;
        case 'o':
            cpu_output_filename = std::string(optarg);
            base_gpu_output_filename = std::string(optarg);
            break;
        default:
            return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
    {
        printf("Error loading source image.\n");
        return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
        "Speedup_noTrf Speedup\n");

    float time_cpu = 0.0;

    /* TODO: run your CPU implementation here and get its time. Don't include
     * file IO in your measurement. Store the time taken in a variable, so
     * it can be printed later for comparison with GPU kernels. */
    /* For example: */
    {
        std::string cpu_file = cpu_output_filename;
        pgm_image cpu_output_img;
        copy_pgm_image_size(&source_img, &cpu_output_img);
        time_cpu = run_best_cpu(FILTER, 3, source_img.matrix, cpu_output_img.matrix, source_img.width,
                                source_img.height); // From kernels.h
        save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
        destroy_pgm_image(&cpu_output_img);
    }

    /* TODO:
     * run each of your gpu implementations here,
     * get their time,
     * and save the output image to a file.
     * Don't forget to add the number of the kernel
     * as a prefix to the output filename:
     * Print the execution times by calling print_run().
     */

    {
        cudaError_t err = cudaDeviceReset(); // Reset device to ensure all kernels start same way
        std::string gpu_file = "1" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        GpuTimes gpu_times = run_kernel1(FILTER, FILTER_DIMENSION, source_img.matrix, gpu_output_img.matrix,
                                         source_img.width, source_img.height);
        print_times(time_cpu, gpu_times);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        cudaDeviceReset(); // Reset device to ensure all kernels start same way
        std::string gpu_file = "2" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        GpuTimes gpu_times = run_kernel2(FILTER, FILTER_DIMENSION, source_img.matrix, gpu_output_img.matrix,
                                         source_img.width, source_img.height);
        print_times(time_cpu, gpu_times);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        cudaDeviceReset(); // Reset device to ensure all kernels start same way
        std::string gpu_file = "3" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        GpuTimes gpu_times = run_kernel3(FILTER, FILTER_DIMENSION, source_img.matrix, gpu_output_img.matrix,
                                         source_img.width, source_img.height);
        print_times(time_cpu, gpu_times);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        cudaDeviceReset(); // Reset device to ensure all kernels start same way
        std::string gpu_file = "4" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        GpuTimes gpu_times = run_kernel4(FILTER, FILTER_DIMENSION, source_img.matrix, gpu_output_img.matrix,
                                         source_img.width, source_img.height);
        print_times(time_cpu, gpu_times);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        cudaDeviceReset(); // Reset device to ensure all kernels start same way
        // We want to ensure pinned memory for faster transfers, so use special versions
        // of pgm functions for this kernel.
        pgm_image pinned_source_img;
        init_pgm_image(&pinned_source_img);
        if (cuda_load_pgm_from_file(input_filename.c_str(), &pinned_source_img) != NO_ERR)
        {
            printf("Kernel 5: Error loading source image.\n");
            return 0;
        }
        std::string gpu_file = "5" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        cuda_copy_pgm_image_size(&pinned_source_img, &gpu_output_img);
        GpuTimes gpu_times = run_kernel5(FILTER, FILTER_DIMENSION, pinned_source_img.matrix, gpu_output_img.matrix,
                                         pinned_source_img.width, pinned_source_img.height);
        print_times(time_cpu, gpu_times);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        cuda_destroy_pgm_image(&gpu_output_img);
    }
}
