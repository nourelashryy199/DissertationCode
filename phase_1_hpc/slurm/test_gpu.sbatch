#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --output=gpu_check_%j.out

nvidia-smi