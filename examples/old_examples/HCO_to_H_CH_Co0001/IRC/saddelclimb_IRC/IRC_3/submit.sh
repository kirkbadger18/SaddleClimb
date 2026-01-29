#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --time=96:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --constraint=48core
#SBATCH --mem=96G
##SBATCH --account=cfgoldsm-condo


source /users/kbadger1/anaconda3/etc/profile.d/conda.sh
conda activate ase

module load hpcx-mpi 
module load quantum-espresso-mpi/7.1-gits-v4bxgtv

python3 run_irc.py

