#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --qos=c2
#SBATCH -t 7-0:00:00
#SBATCH -o jobs_logs/job.%J.out
#SBATCH -e jobs_logs/job.%J.err
#SBATCH --mem=64G

CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/ia2280/.conda/envs/main_env_5
conda activate $CONDA_ENV
python fusion.py