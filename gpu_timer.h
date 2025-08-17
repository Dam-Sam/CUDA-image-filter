typedef struct
{
    int kernel;
    float computation;
    float tx_in;
    float tx_out;
    float total;
    float allocate;
    float free;
} GpuTimes;


class GpuTimer
{
private:
    GpuTimes gpu_times;
    timespec start_time;
    timespec start_compute_time;
    timespec start_tx_in_time;
    timespec start_tx_out_time;
    timespec start_allocate_time;
    timespec start_free_time;
    timespec get_current_time();
    float get_current_duration(timespec start);

public:
    GpuTimer(int kernel);
    void start();
    void stop();
    void start_compute();
    void stop_compute();
    void start_tx_in();
    void stop_tx_in();
    void start_tx_out();
    void stop_tx_out();
    void start_allocate();
    void stop_allocate();
    void start_free();
    void stop_free();
    GpuTimes get_times();
    void print_times();
};
