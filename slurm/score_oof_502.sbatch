#!/bin/bash
#SBATCH --job-name=score_oof502
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/score_oof502_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/score_oof502_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate

NN=/hickory/users/j/o/johnnyta/nnunet
export nnUNet_raw=$NN/nnUNet_raw
export nnUNet_preprocessed=$NN/nnUNet_preprocessed
export nnUNet_results=$NN/nnUNet_results

# 1. Combine the five folds' out-of-fold predictions (204 development cases) and score them
nnUNetv2_accumulate_crossval_results 502 -c 3d_fullres -o $NN/scoring/oof_502_nnunet

# 2. Region-level scores (WT, TC, ET)
python $NN/tools/score_regions.py \
  $nnUNet_raw/Dataset502_BraTSPED_dev/labelsTr \
  $NN/scoring/oof_502_nnunet \
  $NN/scoring/oof_502 \
  "Out-of-fold, 204 development cases (5-fold, nnU-Net 1000 epochs)"
