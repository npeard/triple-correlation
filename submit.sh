#!/bin/bash
#
#
#SBATCH --job-name=triple-corr
#
#SBATCH --time=2-00:00:00 # Maximum runtime (D-HH:mm:SS)
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH -G 1
#SBATCH -C GPU_MEM:80GB
#SBATCH --partition=rotskoff
#
#SBATCH -o output
#SBATCH -e error


module purge
ml python/3.12.1

srun python3 main.py --regenerate_datasets
