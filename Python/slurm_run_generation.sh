#!/bin/bash
#SBATCH --job-name=generate_synthetic_data.py
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=16 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/generate_synthetic_data.err # standard error file
#SBATCH --output=slurm_logs/generate_synthetic_data.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python main_new.py
deactivate
