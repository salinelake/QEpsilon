#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

##SBATCH --mem=64G
##SBATCH --gres=gpu:1
##SBATCH --constraint=gpu80

#SBATCH --time=1:30:00
#SBATCH --job-name=2qubits


# load modules or conda environments here
module purge
module load anaconda3/2023.3
conda activate /home/pinchenx/data.gpfs/envs/qepsilon
python run.py

