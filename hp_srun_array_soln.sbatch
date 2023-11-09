#!/bin/bash
#SBATCH --job-name=hp_array
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=10:00
#SBATCH --mail-type=ALL
#SBATCH --output=run.%A_%a.out
#SBATCH --error=run.%A_%a.err
#SBATCH --array=0-7:4
#SBATCH --ntasks=4

ml purge
ml python/3.9.0
ml py-tensorflow/2.10.0_py39

for i in {0..3}; do
    srun -n 1 python3 hp_tuning.py $((SLURM_ARRAY_TASK_ID+i)) &
done

wait
