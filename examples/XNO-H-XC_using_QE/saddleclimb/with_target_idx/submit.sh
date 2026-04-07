#!/usr/bin/env bash
#SBATCH --job-name=saddleclimb
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --constraint=32core
#SBATCH --mem=96G
##SBATCH --account=cfgoldsm-condo


source /users/kbadger1/anaconda3/etc/profile.d/conda.sh
conda activate ase

module load hpcx-mpi 
module load quantum-espresso-mpi/7.1-5jax

python3 run_climb.py
