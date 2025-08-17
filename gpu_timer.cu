#include "gpu_timer.h"
#include <iostream>

timespec GpuTimer::get_current_time()
{
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    return current_time;
}

float GpuTimer::get_current_duration(timespec start)
{
    struct timespec stop_time = get_current_time();
    return (stop_time.tv_sec - start.tv_sec) + (stop_time.tv_nsec - start.tv_nsec) / 1e9;
}


GpuTimer::GpuTimer(int kernel)
{
    gpu_times.kernel = kernel;
}

void GpuTimer::start()
{
    start_time = get_current_time();
}

void GpuTimer::stop()
{
    gpu_times.total = get_current_duration(start_time);
}

void GpuTimer::start_compute()
{
    start_compute_time = get_current_time();
}

void GpuTimer::stop_compute()
{
    gpu_times.computation = get_current_duration(start_compute_time);
}


void GpuTimer::start_tx_in()
{
    start_tx_in_time = get_current_time();
}

void GpuTimer::stop_tx_in()
{
    gpu_times.tx_in = get_current_duration(start_tx_in_time);
}

void GpuTimer::start_tx_out()
{
    start_tx_out_time = get_current_time();
}

void GpuTimer::stop_tx_out()
{
    gpu_times.tx_out = get_current_duration(start_tx_out_time);
}

void GpuTimer::start_allocate()
{
    start_allocate_time = get_current_time();
}

void GpuTimer::stop_allocate()
{
    gpu_times.allocate = get_current_duration(start_allocate_time);
}

void GpuTimer::start_free()
{
    start_free_time = get_current_time();
}

void GpuTimer::stop_free()
{
    gpu_times.free = get_current_duration(start_free_time);
}

GpuTimes GpuTimer::get_times()
{
    return gpu_times;
}

void GpuTimer::print_times()
{
    std::cout << "GPU Times for Kernel " << gpu_times.kernel << std::endl;
    std::cout << "Alloc: \t" << gpu_times.allocate << std::endl;
    std::cout << "TXin: \t" << gpu_times.tx_in << std::endl;
    std::cout << "Comp: \t" << gpu_times.computation << std::endl;
    std::cout << "TXOut: \t" << gpu_times.tx_out << std::endl;
    std::cout << "Free: \t" << gpu_times.free << std::endl;
    std::cout << "Total: \t" << gpu_times.total << std::endl;
}
