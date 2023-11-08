#!/bin/bash

#SBATCH --account=PHYS030544 
#SBATCH --job-name=Vector_LL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=0:30:00
#SBATCH --partition=test
#SBATCH --array=1-10

echo "Running tests"
echo "Before activation python =  $(which python)"

module load languages/miniconda

source activate LL_env 
echo "After activation python = $(which python)"

module load languages/intel/2020-u4

size=$((SLURM_ARRAY_TASK_ID*100))

srun python LL.py 50 $size 0.5 0

srun --mpi=pmi2 python LL_MPI.py 50 $size 0.5 0

srun --mpi=pmi2 python LL_MPI_Vec.py 50 $size 0.5 0

srun --mpi=pmi2 python LL_MPI_Vec_Jit.py 50 $size 0.5 0

