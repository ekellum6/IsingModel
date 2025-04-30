#!/bin/bash
###############################################################################
# Ising3d_CUDA.sh
#
# * requests 1 NVIDIA GPU of *any* type            (change if you need A100 etc.)
# * compiles   MCS_Test_CUDA.cu  with nvcc
# * runs the executable with srun
###############################################################################
#SBATCH -J ising3d-cuda
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o log_%j.out
#SBATCH --mail-type=FAIL

# ----- modules --------------------------------------------------------------
module purge
module load gcc
module load cuda

# ----- build ---------------------------------------------------------------
nvcc -std=c++17 -O3 -arch=sm_70 MCS_Test_CUDA.cu -o MCS_Test_CUDA

# ----- run -----------------------------------------------------------------
srun ./MCS_Test_CUDA
