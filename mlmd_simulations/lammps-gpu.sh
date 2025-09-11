#!/bin/bash -l
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --account=project_462000738
#SBATCH --output=test_lammps-gpu_%j.out
#SBATCH --error=test_lammps-gpu_%j.err
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH --hint=nomultithread

module purge
module load LUMI/24.03
module load LAMMPS/stable-2Aug2023-update4-pair-allegro-rocm-6.0.3-pytorch-2.3.1-20241203

# Enable GPU-aware MPI support
export MPICH_GPU_SUPPORT_ENABLED=1

# Set threading environment variables
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


#CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

srun -n 8 lmp -sf kk -k on g 8 -pk kokkos gpu/aware off newton on neigh full -in 2xe_cc3_tba_lammps.in
