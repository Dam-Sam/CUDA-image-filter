#!/bin/bash

make
./test_solution
echo "2x2"
./main -i input_2x2.pgm -o output_2x2.pgm
echo "64x64"
./main -i input64.pgm -o output64.pgm
echo "500x500"
./main -i input_raw.pgm -o output_raw.pgm
echo "~2500x1800"
./main -i input_2500.pgm -o output_2500.pgm
echo "~5000x3500"
./main -i input_5kx5k.pgm -o output_5kx5k.pgm
echo "10MB (3200x3200)"
./main -i input_10mb.pgm -o output_10mb.pgm

