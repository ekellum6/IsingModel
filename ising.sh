#!/bin/bash
#SBATCH -J ising-pool
#SBATCH -N1 --cpus-per-task=40       # 40 CPUs on one node
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o pool_%j.out

module load anaconda3

srun python ising_driver.py --dim 3 --L 100 --tmin 1.0 --tmax 8.0 --nT 200 --steps 7000 --eq 3000 --outfile "results.csv"