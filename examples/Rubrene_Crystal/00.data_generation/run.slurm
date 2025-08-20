#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 5:00:0
#SBATCH --gres=gpu:1
#SBATCH --job-name=q_transport
#SBATCH --array=0-7
 
module load anaconda3/2021.11
conda activate qepsilon

run_id=$SLURM_ARRAY_TASK_ID
python -u 01.run_quantum.py --temperature 200.0 --epsilon 0.1 --dt 0.01 --sample_time 100 --batchsize 8 --run_id $run_id 
python -u 01.run_quantum.py --temperature 250.0 --epsilon 0.1 --dt 0.01 --sample_time 100 --batchsize 8 --run_id $run_id 
python -u 01.run_quantum.py --temperature 300.0 --epsilon 0.1 --dt 0.01 --sample_time 100 --batchsize 8 --run_id $run_id 
python -u 01.run_quantum.py --temperature 350.0 --epsilon 0.1 --dt 0.01 --sample_time 100 --batchsize 8 --run_id $run_id 
python -u 01.run_quantum.py --temperature 400.0 --epsilon 0.1 --dt 0.01 --sample_time 100 --batchsize 8 --run_id $run_id 
