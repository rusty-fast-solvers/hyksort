#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --distribution=block:block
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1

module load PrgEnv-aocc
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
module swap craype-network-ofi craype-network-ucx
module swap cray-mpich cray-mpich-ucx
export UCX_IB_REG_METHODS=direct

# check which library versions the executable is pointing too
ldd hyksort

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/alltoall/
# Set simulation parameters

# Create a CSV output file for analysis
export OUTPUT=${SLURM_JOBID}_${SLURM_JOB_NUM_NODES}_${SLURM_TASKS_PER_NODE}.csv
touch ${OUTPUT}
echo "n_processes,nparticles_per_core,kway,intrinsic" >> ${OUTPUT}

# Run jobs
srun -n $SLURM_NTASKS --hint=nomultithread ${TEST}/hyksort >> ${OUTPUT}
