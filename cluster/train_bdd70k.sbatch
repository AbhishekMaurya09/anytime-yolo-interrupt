#!/bin/bash
# SLURM job: train the anytime detector on the ~67k-image BDD100K split.
# Submit from ~/anytime_yolo_cs25m201:   sbatch cluster/train_bdd70k.sbatch
# Resubmit the same command after a time-limit stop; it resumes from last.pt.
#SBATCH --job-name=anytime70k
#SBATCH --partition=q3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd ~/anytime_yolo_cs25m201
mkdir -p logs

RUN=runs/anytime_interrupt/train6_bdd70k
DATA=data/bdd70k
EPOCHS=${EPOCHS:-30}

if [ -f "$RUN/last.pt" ]; then
    START=(--resume "$RUN/last.pt")
else
    START=(--weights runs/anytime_interrupt/train5/last.pt)
fi

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, start: ${START[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# batch 16 / lr 0.004 keeps lr-per-image equal to the laptop runs (batch 4 / lr 0.001)
env/bin/python -u -m anytime_yolo_interrupt.scripts.train \
    --data-root "$DATA" --data-yaml "$DATA/data.yaml" \
    --epochs "$EPOCHS" --close-mosaic 3 --batch-size 16 --lr 0.004 \
    --img-size 640 --device cuda --val-every 2 --workers 8 \
    --log-dir "$RUN" "${START[@]}"
