#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --time=48:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=32
#SBATCH --constraint=32core
#SBATCH --mem=96G
##SBATCH --account=cfgoldsm-condo


source /users/kbadger1/anaconda3/etc/profile.d/conda.sh
module load hpcx-mpi 
module load quantum-espresso-mpi/7.3s-kydgjwo
conda activate ase


python3 run_climb.py
#mpirun -n 96 --bind-to core /oscar/runtime/software/external/quantum-espresso/7.1-git/bin/pw.x 
