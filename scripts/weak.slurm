#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 64

# Restore AMD compiler env
# module restore PrgEnv-aocc

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/alltoall/
export SCRATCH=${TEST}/${SLURM_JOBID}

# Create a scratch directory for this run
mkdir -p ${SCRATCH}
cd ${SCRATCH}
# Set simulation parameters

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "n_processes,kway,intrinsic" >> ${OUTPUT}

# Run jobs
srun --n $SLURM_NTASKS --hint=nomultithread ${TEST}/hyksort >> ${OUTPUT}
