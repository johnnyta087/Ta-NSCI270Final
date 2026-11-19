#!/bin/bash
#SBATCH --job-name=nnunet_plan502
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/plan502_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/plan502_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate

export nnUNet_raw=/hickory/users/j/o/johnnyta/nnunet/nnUNet_raw
export nnUNet_preprocessed=/hickory/users/j/o/johnnyta/nnunet/nnUNet_preprocessed
export nnUNet_results=/hickory/users/j/o/johnnyta/nnunet/nnUNet_results

nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres --verify_dataset_integrity -np 8
