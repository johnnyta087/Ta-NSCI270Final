#!/bin/bash
#SBATCH --job-name=nnunet_smoke501
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/smoke501_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/smoke501_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate

export nnUNet_raw=/hickory/users/j/o/johnnyta/nnunet/nnUNet_raw
export nnUNet_preprocessed=/hickory/users/j/o/johnnyta/nnunet/nnUNet_preprocessed
export nnUNet_results=/hickory/users/j/o/johnnyta/nnunet/nnUNet_results
export nnUNet_n_proc_DA=8

nvidia-smi
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_5epochs
