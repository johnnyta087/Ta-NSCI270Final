#!/bin/bash
#SBATCH --job-name=viz502
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hickory/users/j/o/johnnyta/nnunet/logs/viz502_%j.out
#SBATCH --error=/hickory/users/j/o/johnnyta/nnunet/logs/viz502_%j.err

set -euo pipefail
module load python/3.12.4
source /hickory/users/j/o/johnnyta/nnunet/env/bin/activate
python /hickory/users/j/o/johnnyta/nnunet/tools/build_results_viz.py
