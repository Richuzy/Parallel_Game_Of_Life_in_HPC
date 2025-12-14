#!/bin/bash

#SBATCH -A account_name
#SBATCH -p standard
#SBATCH -J gol_project_multi_gun
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gol

export OMPI_MCA_btl=^openib

mpirun -np $SLURM_NTASKS python -m mpi4py gol_toy_multi_gun.py \
    --nx 16384 --ny 16384 --steps 2000 \
    --guns "10,10;16000,16000" --viz_every 0
