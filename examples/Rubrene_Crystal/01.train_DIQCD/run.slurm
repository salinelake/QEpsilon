#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5:00:0
#SBATCH --gres=gpu:1
#SBATCH --job-name=q_transport
#SBATCH --array=0-4
 
module load anaconda3/2021.11
conda activate qepsilon

temperature_list=(200 250 300 350 400)
temperature=${temperature_list[$SLURM_ARRAY_TASK_ID]}
python train.py --temperature $temperature
