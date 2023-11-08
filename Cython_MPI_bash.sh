#!/bin/bash

# Define the number of times you want to run the MPI Cython script
num_runs=10

# Define the starting value for nmax
start_nmax=100

# Define the step size for nmax
nmax_step=100

# Define the available number of processors
available_processors=(1 2 3 4 5 6)

# Loop through the desired number of runs
for ((i=1; i<=$num_runs; i++))
do
    # Calculate the current nmax value
    current_nmax=$((start_nmax + (i-1) * nmax_step))

    # Loop through the available number of processors
    for num_processes in "${available_processors[@]}"
    do
        # Run the MPI Cython script with mpirun
        mpiexec -n $num_processes python3 MPI_Cython_LL.py 50 $current_nmax 0.5 0 >> mpi_outpu.out
    done
done