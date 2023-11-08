#!/bin/bash


#SBATCH --account=PHYS030544 
#SBATCH --job-name=Vector_LL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=800
#SBATCH --time=0:05:00
#SBATCH --partition=test
#SBATCH --array=1-10

echo "Running tests"
echo "Before activation python =  $(which python)"

module load languages/miniconda

source activate LL_env 
echo "After activation python = $(which python)"

size=$((SLURM_ARRAY_TASK_ID*25))

python Vectorized_LebwohlLasher.py 50 $size 0.5 0

