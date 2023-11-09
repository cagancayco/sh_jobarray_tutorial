#!/bin/bash
#SBATCH --job-name=hp_array
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=10:00
#SBATCH --mail-type=ALL
#SBATCH --output=run.%A_%a.out
#SBATCH --error=run.%A_%a.err
#SBATCH --array=0-1

ml purge
ml python/3.9.0
ml py-tensorflow/2.10.0_py39

python3 hp_tuning.py ${SLURM_ARRAY_TASK_ID}
