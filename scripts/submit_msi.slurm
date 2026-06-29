#!/usr/bin/env bash
#SBATCH --job-name=geosam3d
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/geosam3d_%j.out
#SBATCH --error=logs/geosam3d_%j.err
set -euo pipefail
# Known-good MSI GPU env (a100-4). One-time setup, see scripts/MSI_RUNBOOK.md:
#   conda create -n geosam python=3.11 && source activate geosam && pip install -e .
#   then install MonoGS (sibling pkg) + SAM 2 weights for the real recon path.
module load gcc/13.1.0-5z64cho
module load cuda/12.4
source activate ${PROJ_ENV:-geosam}
unset LD_LIBRARY_PATH   # MSI modules clobber LD_LIBRARY_PATH and break conda CUDA libs
# v100 fallback (faster queue): --partition=v100 --gres=gpu:v100:1

EXP=${1:-default}
OUT=outputs/${EXP}_${SLURM_JOB_ID}
mkdir -p logs "$OUT"

srun python -m geosam3d.training.train --config-name $EXP train.out_dir=$OUT
