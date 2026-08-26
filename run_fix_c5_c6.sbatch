#!/bin/bash
#SBATCH -J fix_c5_c6
#SBATCH -p main-cpu                  # CPU-only partition, open to all users
#SBATCH -w caserta                   # 192 cores / 2TB: the only main-cpu node
                                     # with enough RAM for 60 workers
#SBATCH -c 60                        # 60 cores on one node = 60 workers
#SBATCH --mem=480G                   # 60 workers x ~7GB + headroom
#SBATCH --time=2-00:00               # 2 days (D-HH:MM)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s2850680@student.utwente.nl

echo "node          : $(hostname)"
echo "cores         : $SLURM_CPUS_PER_TASK"
echo "started       : $(date)"

module load miniconda3/25.7
source activate fft

cd "$SLURM_SUBMIT_DIR"

python ./FFT_simulation/batch_run.py \
    ./FFT_simulation/Run_configs/fix_c5_c6_gmres400.yaml \
    --max-workers 60

echo "finished      : $(date)"
