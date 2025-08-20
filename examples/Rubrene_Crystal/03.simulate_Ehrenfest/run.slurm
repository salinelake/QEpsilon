#!/bin/bash
#SBATCH --job-name=transport        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH --qos=premium
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-4

source ~/.bashrc
conda activate qepsilon

temperature_list=(200 250 300 350 400)
temperature=${temperature_list[$SLURM_ARRAY_TASK_ID]}
python 01.simulate.py --temperature $temperature
