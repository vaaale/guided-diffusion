#!/bin/bash
#SBATCH --job-name=DebugSlurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=linux[1-2]

#srun mkdir saved_models
srun run_slurm.sh

