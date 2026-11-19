#!/bin/bash
#SBATCH --job-name=rescore_viz502
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/rescore_viz502_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/rescore_viz502_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate
NN=/hickory/users/j/o/johnnyta/nnunet

python $NN/tools/score_regions.py $NN/nnUNet_raw/Dataset502_BraTSPED_dev/labelsTr $NN/scoring/oof_502_nnunet $NN/scoring/oof_502 "Out-of-fold, 204 development cases (5-fold, nnU-Net 1000 epochs)"
python $NN/tools/score_regions.py $NN/holdout_test/labels $NN/scoring/holdout_502_pred $NN/scoring/holdout_502 "Held-out test set, 51 cases (5-fold ensemble, scored once)"
python $NN/tools/build_results_viz.py
