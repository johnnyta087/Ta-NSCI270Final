#!/bin/bash
#SBATCH -J skullstrip_val
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH -t 3:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o skullstrip_val_%j.out
#SBATCH -e skullstrip_val_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=johnnyta@unc.edu

# ---- EDIT THESE ----
ENV_PATH=/hickory/users/j/o/johnnyta/envs/hdbet
IN_ROOT="/hickory/users/j/o/johnnyta/PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Validation"
OUT_ROOT=/hickory/users/j/o/johnnyta/brats-peds-stripped/Validation
# --------------------

module purge
module load anaconda/2024.02
conda activate "$ENV_PATH"

nvidia-smi
python skullstrip.py "$IN_ROOT" "$OUT_ROOT"
