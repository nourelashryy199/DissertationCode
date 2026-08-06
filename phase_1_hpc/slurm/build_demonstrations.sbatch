#!/bin/bash
#SBATCH --job-name=build_demos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=build_demos_%j.out
#SBATCH --error=build_demos_%j.err

# NOTE: module name is a placeholder — verify with `module avail python`
# or `module avail anaconda` on Stanage and correct before submitting.
module load Anaconda3/2024.02-1
source activate dissertation_env

export DISSERTATION_ROOT=/mnt/parscratch/users/$USER/dissertation/phase_1_hpc
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd $DISSERTATION_ROOT

echo "Starting build_demonstrations.py at $(date)"
python scripts/build_demonstrations.py
echo "Finished at $(date)"