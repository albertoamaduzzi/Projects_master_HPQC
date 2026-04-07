#!/bin/bash
# Submit the full benchmark (CPU + GPU) to a node that has an A100 GPU.
#
# Usage (from the scripts directory):
#   cd /path/to/scripts
#   sbatch submit_gpu_pipeline.sh
#
# The GPU runs (section 3d in main.py) require a real A100 device; all CPU
# runs (none, just_omp, omp_mpi) work identically on GPU nodes.
# Results are appended to the same CSV as the CPU pipeline, so plots from
# analysis.py will show all four cases side-by-side.

#SBATCH --job-name=jacobi_gpu_pipeline
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=tra26_mhpcqc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load gcc openmpi cmake python nvhpc
source my_science_env/bin/activate
python3 main.py
