#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=00:30:00
#SBATCH --nodes=1

# Restore AMD compiler env
# module restore PrgEnv-aocc

# Export PKG_CONFIG path, missing on Archer2
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/cray/pe/mpich/8.0.16/ofi/gnu/9.1/lib/pkgconfig

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/alltoall/
export SCRATCH=${TEST}/${SLURM_JOBID}

# Create a scratch directory for this run
mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Set simulation parameters
n_tasks=(64)


# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "n_processes,kway,intrinsic" >> ${OUTPUT}

# Run jobs
for i in ${!n_tasks[@]}; do
    srun --ntasks=${n_tasks[$i]} --ntasks-per-core=1 --nodes=1 ${TEST}/hyksort >> ${OUTPUT}
done

