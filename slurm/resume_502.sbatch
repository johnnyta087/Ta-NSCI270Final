#!/bin/bash
#SBATCH --job-name=nnunet_resume502
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/resume502_%j_fold${FOLD}.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/resume502_%j_fold${FOLD}.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate

export nnUNet_raw=/hickory/users/j/o/johnnyta/nnunet/nnUNet_raw
export nnUNet_preprocessed=/hickory/users/j/o/johnnyta/nnunet/nnUNet_preprocessed
export nnUNet_results=/hickory/users/j/o/johnnyta/nnunet/nnUNet_results
export nnUNet_n_proc_DA=8

echo "resuming fold ${FOLD} on $(hostname)"
nvidia-smi
nnUNetv2_train 502 3d_fullres "${FOLD}" --c
