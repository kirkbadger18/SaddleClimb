#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --constraint=48core
#SBATCH --mem=96G
#SBATCH --account=cfgoldsm-condo


module load anaconda/2023.09-0-7nso27y
source /oscar/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh

module load vasp-mpi/6.4.2_cfgoldsm-7krhcss

conda activate ase

export ASE_VASP_VDW=/oscar/runtime/software/external/vasp/6.4.2_cfgoldsm/source/
export VASP_PP_PATH=/oscar/runtime/software/external/vasp/6.4.2_cfgoldsm/source/
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_STACKSIZE=512m
export OMP_NUM_THREADS=1
export ASE_VASP_COMMAND="mpirun -np 48 --bind-to core vasp_std"

#mpirun -np 48 --bind-to core vasp_std
python3 run_opt.py
