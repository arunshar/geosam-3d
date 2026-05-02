#!/usr/bin/env bash
#SBATCH --job-name=geosam3d
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/geosam3d_%j.out
#SBATCH --error=logs/geosam3d_%j.err
set -euo pipefail
module load python/3.11
module load cuda/12.4
source ~/envs/geosam3d/bin/activate

EXP=${1:-default}
OUT=outputs/${EXP}_${SLURM_JOB_ID}
mkdir -p logs "$OUT"

srun python -m geosam3d.training.train --config-name $EXP train.out_dir=$OUT
