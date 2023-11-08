#!/bin/bash

# Define the number of times you want to run the Python script
num_runs=1

# Define the starting value for nmax
start_nmax=100

# Define the step size for nmax
nmax_step=100

# Loop through the desired number of runs
for ((i=1; i<=$num_runs; i++))
do
    #current_nmax=$((start_nmax + (i-1) * nmax_step))
    # Run the Python script and append the output to a text file
    python3 .py 50 $num_runs*100 0.5 0 >> "output2.out"
done