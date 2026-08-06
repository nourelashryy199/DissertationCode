#!/bin/bash
#SBATCH --job-name=stage_a_qwen14b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=64G
#SBATCH --output=stage_a_qwen14b_%j.out
#SBATCH --error=stage_a_qwen14b_%j.err

PYTHON=/users/msp25noe/.conda/envs/dissertation_env/bin/python3

$PYTHON --version
$PYTHON -c "import sys; assert sys.version_info >= (3, 10), f'Python too old: {sys.version}'"

export DISSERTATION_ROOT=/users/msp25noe/Thesis_Code/DissertationCode/phase_1_hpc
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd $DISSERTATION_ROOT

echo "Starting run_stage_a.py at $(date)"
$PYTHON scripts/run_stage_a.py --model Qwen/Qwen2.5-14B-Instruct --sample_size 45
echo "Finished at $(date)"
