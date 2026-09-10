#!/bin/bash
#SBATCH -J cnn_train
#SBATCH -p main-gpu                  # CHECK: your GPU partition name
#SBATCH --gres=gpu:4                 # 4 GPUs on one node
#SBATCH -c 8                         # dataloader workers + main process
#SBATCH --mem=64G
#SBATCH --time=0-08:00               # D-HH:MM
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s2850680@student.utwente.nl

# Two ways to use the node - pick one, they are mutually exclusive.
#
#   MODE=ddp    one model split over all 4 GPUs (fast per epoch, but with
#               ~100 training cases the sync overhead dominates)
#   MODE=sweep  four independent models, one per GPU, different split seeds
#               (recommended at this dataset size: same wall time, and the
#               spread across seeds tells you how solid the score really is)
MODE=${MODE:-sweep}

echo "node          : $(hostname)"
echo "gpus          : $CUDA_VISIBLE_DEVICES"
echo "mode          : $MODE"
echo "started       : $(date)"

module load miniconda3/25.7
source activate fft

cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Build the case index if it is not there yet (cheap, CPU only).
if [ ! -f CNN/dataset_index.csv ]; then
    python CNN/build_dataset.py
fi

if [ "$MODE" = "ddp" ]; then
    NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
    torchrun --standalone --nproc_per_node="$NGPU" \
        CNN/train.py --epochs 200 --batch-size 4
else
    python CNN/sweep.py --gpus 0,1,2,3 --seeds 0,1,2,3 -- --epochs 200
fi

echo "finished      : $(date)"
