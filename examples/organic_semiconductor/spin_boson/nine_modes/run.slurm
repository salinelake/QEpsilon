#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 5:00:0
#SBATCH --gres=gpu:1
#SBATCH --job-name=nine_modes
  
module load anaconda3/2021.11
conda activate qepsilon
python run_multi_mode.py
