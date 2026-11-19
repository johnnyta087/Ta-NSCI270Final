#!/bin/bash
#SBATCH --job-name=score_holdout502
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/score_holdout502_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/score_holdout502_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate

NN=/hickory/users/j/o/johnnyta/nnunet
export nnUNet_raw=$NN/nnUNet_raw
export nnUNet_preprocessed=$NN/nnUNet_preprocessed
export nnUNet_results=$NN/nnUNet_results
export nnUNet_n_proc_DA=8

nvidia-smi

# 1. Single prediction run on the 51 held-out test cases, five-fold ensemble (never used in training)
nnUNetv2_predict -i $NN/holdout_test/images -o $NN/scoring/holdout_502_pred \
  -d 502 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer -p nnUNetPlans

# 2. nnU-Net per-label scores
nnUNetv2_evaluate_folder $NN/holdout_test/labels $NN/scoring/holdout_502_pred \
  -djfile $nnUNet_raw/Dataset502_BraTSPED_dev/dataset.json \
  -pfile $nnUNet_preprocessed/Dataset502_BraTSPED_dev/nnUNetPlans.json \
  -o $NN/scoring/holdout_502_pred/summary.json

# 3. Region-level scores (WT, TC, ET)
python $NN/tools/score_regions.py \
  $NN/holdout_test/labels \
  $NN/scoring/holdout_502_pred \
  $NN/scoring/holdout_502 \
  "Held-out test set, 51 cases (5-fold ensemble, scored once)"
