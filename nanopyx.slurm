#!/bin/bash

# General Info: 
# Nanopyx requires 150GB for 1000 frames of temporal correlation
# About 2-3 minutes per eSRRF at 250 frames of correlation
# About 25 GB for 250 frames of temporal correlation 



#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=02:00:00
#SBATCH --job-name=Nanopyx
#SBATCH --mem=60G 
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=jake.ankrum@embl.de
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:1
#SBATCH -o ./Reports/out/slurm.%N.%j.out
#SBATCH -e ./Reports/err/slurm.%N.%j.err

# Exit on errors
set -e 

# Set required modules
module purge
module load CUDA/12.1.1

# Navigate to working directory
cd /scratch/ankrum/henrique_nanopyx

echo "Activating environment... "
source ./activate.sh

echo "Running Script..."
python comp_analysis.py 

echo "Job successful!"
