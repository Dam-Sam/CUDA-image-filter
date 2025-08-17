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
#include "pgm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define MAX_THREADS 8
#define CHUNK_SIZE 32
#define QUEUE_SIZE 1000

typedef struct
{
    int start_x, start_y;
    int width, height;
    const int8_t *filter;
    int filter_dim;
    const int32_t *input;
    int32_t *output;
    int img_width, img_height;
} Tile;

Tile work_queue[QUEUE_SIZE];
int queue_size = 0; // Total tiles in the queue
pthread_mutex_t queue_mutex;
pthread_cond_t queue_cond;

int processing_complete = 0; // Global flag

void find_min_max_cpu(const int32_t *data, int32_t size, int32_t &min_val, int32_t &max_val)
{
    min_val = INT32_MAX;
    max_val = INT32_MIN;
    for (int i = 0; i < size; ++i)
    {
        if (data[i] < min_val)
            min_val = data[i];
        if (data[i] > max_val)
            max_val = data[i];
    }
}

void normalize_cpu(int32_t *image, int32_t width, int32_t height, int32_t min, int32_t max)
{
    int32_t num_pixels = width * height;
    float range = max - min;

    // Avoid division by zero if max equals min
    if (range == 0)
    {
        range = 1;
    }

    for (int32_t i = 0; i < num_pixels; i++)
    {
        float normalized = 255.0f * (image[i] - min) / range;
        image[i] = (int32_t)(fmax(0.0f, fmin(255.0f, normalized)));
    }
}

void process_tile(Tile *tile)
{
    for (int y = 0; y < tile->height; y++)
    {
        for (int x = 0; x < tile->width; x++)
        {
            int sum = 0;
            for (int fy = 0; fy < tile->filter_dim; fy++)
            {
                for (int fx = 0; fx < tile->filter_dim; fx++)
                {
                    int imgX = tile->start_x + x + fx - tile->filter_dim / 2;
                    int imgY = tile->start_y + y + fy - tile->filter_dim / 2;

                    if (imgX >= 0 && imgX < tile->img_width && imgY >= 0 && imgY < tile->img_height)
                    {
                        sum += tile->input[imgY * tile->img_width + imgX] * tile->filter[fy * tile->filter_dim + fx];
                    }
                }
            }
            tile->output[(tile->start_y + y) * tile->img_width + (tile->start_x + x)] = sum;
        }
    }
}

void *worker_thread(void *arg)
{
    while (1)
    {
        pthread_mutex_lock(&queue_mutex);

        while (queue_size == 0 && processing_complete == 0)
        {
            pthread_cond_wait(&queue_cond, &queue_mutex);
        }

        if (queue_size == 0 && processing_complete == 1)
        {
            pthread_mutex_unlock(&queue_mutex);
            break;
        }

        Tile tile = work_queue[--queue_size];
        if (queue_size == 0)
        {
            processing_complete = 1;
            pthread_cond_broadcast(&queue_cond); // Signal all threads to exit
        }

        pthread_mutex_unlock(&queue_mutex);

        process_tile(&tile);
    }

    return NULL;
}

void fillWorkQueue(const int8_t *filter, int32_t dimension, const int32_t *input,
                   int32_t *output, int32_t width, int32_t height)
{
    int tiles_x = ceil((float)width / CHUNK_SIZE);
    int tiles_y = ceil((float)height / CHUNK_SIZE);

    for (int x = 0; x < tiles_x; x++)
    {
        for (int y = 0; y < tiles_y; y++)
        {
            Tile tile;
            tile.start_x = x * CHUNK_SIZE;
            tile.start_y = y * CHUNK_SIZE;
            tile.width = (tile.start_x + CHUNK_SIZE > width) ? (width - tile.start_x) : CHUNK_SIZE;
            tile.height = (tile.start_y + CHUNK_SIZE > height) ? (height - tile.start_y) : CHUNK_SIZE;

            tile.filter = filter;
            tile.filter_dim = dimension;
            tile.input = input;
            tile.output = output;
            tile.img_width = width;
            tile.img_height = height;

            pthread_mutex_lock(&queue_mutex);
            if (queue_size < QUEUE_SIZE)
            {
                work_queue[queue_size++] = tile;
            }
            pthread_mutex_unlock(&queue_mutex);
        }
    }
}

double run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                    int32_t *output, int32_t width, int32_t height)
{

    struct timespec start, stop;

    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_t threads[MAX_THREADS];
    pthread_mutex_init(&queue_mutex, NULL);
    pthread_cond_init(&queue_cond, NULL);

    fillWorkQueue(filter, dimension, input, output, width, height);

    // Create worker threads
    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, worker_thread, NULL);
    }

    // Wait for all threads to finish
    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    pthread_mutex_destroy(&queue_mutex);
    pthread_cond_destroy(&queue_cond);

    int32_t h_min, h_max;
    find_min_max_cpu(output, width * height, h_min, h_max);
    normalize_cpu(output, width, height, h_min, h_max);

    clock_gettime(CLOCK_MONOTONIC, &stop);

    return (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1e9;
}
