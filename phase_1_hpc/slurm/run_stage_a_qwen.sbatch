#!/bin/bash
#SBATCH --job-name=stage_a_qwen7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=64G
#SBATCH --output=stage_a_qwen7b_%j.out
#SBATCH --error=stage_a_qwen7b_%j.err

module load Anaconda3/2024.02-1
source activate dissertation_env

export DISSERTATION_ROOT=/mnt/parscratch/users/$USER/dissertation/phase_1_hpc
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd $DISSERTATION_ROOT

echo "Starting run_stage_a.py at $(date)"
python scripts/run_stage_a.py --model Qwen/Qwen2.5-7B-Instruct
echo "Finished at $(date)"