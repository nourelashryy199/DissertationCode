#!/bin/bash
# ============================================================
# phase_0/slurm/smoke_test.sbatch
# Quick GPU job for smoke_test_generation.py — 3 generations
# total, should finish in well under 5 minutes of actual compute.
#
# FILL IN the #SBATCH lines marked TODO below by copying the
# matching lines from your existing
# phase_1_hpc/slurm/run_stage_a_qwen7b.sbatch — partition,
# account, and module-load lines are cluster-account-specific
# and shouldn't be guessed here.
# ============================================================

#SBATCH --job-name=smoke_test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=smoke_test_%j.out
#SBATCH --error=smoke_test_%j.err
# TODO: --partition=<copy from run_stage_a_qwen7b.sbatch>
# TODO: --account=<copy from run_stage_a_qwen7b.sbatch>
# TODO: --mem=<copy from run_stage_a_qwen7b.sbatch>

# TODO: copy your module load / conda activate / venv activate
# lines here, exactly as in run_stage_a_qwen7b.sbatch

export DISSERTATION_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")/../phase_1_hpc
# ^ resolves to the phase_1_hpc/ directory regardless of where
#   this job is submitted from — adjust if your repo layout
#   differs from phase_0/ and phase_1_hpc/ as siblings.

cd "$(dirname "$(readlink -f "$0")")/../scripts"
python smoke_test_generation.py --model Qwen/Qwen2.5-7B-Instruct
