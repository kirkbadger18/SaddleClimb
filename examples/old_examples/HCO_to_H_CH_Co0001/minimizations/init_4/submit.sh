#!/usr/bin/env bash
#SBATCH --job-name=HXCO
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --constraint=48core
#SBATCH --mem=96G
##SBATCH --account=cfgoldsm-condo


source /users/kbadger1/anaconda3/etc/profile.d/conda.sh
conda activate ase

module load hpcx-mpi 
module load quantum-espresso-mpi/7.1-gits-v4bxgtv

python3 run_opt.py

