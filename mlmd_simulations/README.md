# Machine Learning Molecular Dynamics (MLMD) Simulations: Production MLMD-5 Simulation

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the input files and job scripts for performing the fifth round of machine learning molecular dynamics simulations (MLMD-5) using MLIP-2 for Xe@CC3@TBA.

## Simulation Protocol

### Three-Stage Simulation Design

#### Stage 1: Energy Minimization
- **Method**: Conjugate gradient minimization
- **Convergence criteria**: 
  - Energy tolerance: 1.0×10<sup>-4</sup>
  - Force tolerance: 1.0×10<sup>-6</sup>
- **Maximum steps**: 10 000 iterations

#### Stage 2: Equilibration (100 ps)
- **Duration**: 100 ps (100 000 timesteps)
- **Temperature**: 300 K (constant)
- **Ensemble**: NVT with Nosé-Hoover thermostat
- **Thermostat damping**: 100×dt = 0.1 ps
- **Initial velocities**: Maxwell-Boltzmann distribution at 300 K

#### Stage 3: Production Run (1.1 ns)
- **Duration**: 1.1 ns (1 100 000 timesteps)
- **Temperature**: 300 K (constant)
- **Ensemble**: NVT with Nosé-Hoover thermostat
- **Sampling frequency**: Every 100 timesteps (0.1 ps intervals)

## System Specifications

### Molecular System
- **System**: 2Xe_CC3_TBA (MLMD-5 simulations)
- **Total atoms**: 1170 atoms
- **Periodic boundary conditions**: 23.865 Å × 23.865 Å × 23.865 Å

### Simulation Parameters
```
timestep: 1.0 fs
temperature: 300 K
total_simulation_time: 1.1 ns (0.1 ns equil + 1.1 ns prod)
sampling_interval: 0.1 ps
total_trajectory_frames: 11 000
```

## Input Files

### LAMMPS Input Script ([`2xe_cc3_tba_lammps.in`](./2xe_cc3_tba_lammps.in))

#### Force Field Integration
```lammps
# Allegro potential loaded automatically via pair_style allegro
pair_style allegro
pair_coeff * * 2xe_cc3_tba_new-deployed_vf.pth O C H F Xe N
```

#### Thermodynamic Output
```lammps
thermo_style custom step temp press pe ke etotal vol
thermo 100  # Output every 100 steps
```

#### Trajectory Output
```lammps
# Equilibration trajectory
dump equil_dump all custom 100 equilibration.dump id type x y z vx vy vz fx fy fz

# Production trajectory  
dump prod_dump all custom 100 production.dump id type x y z vx vy vz fx fy fz
```

### Initial Structure ([`2xe_cc3_tba_str.data`](./2xe_cc3_tba_str.data))

#### Atomic Masses
```
1 15.9994   # O (Oxygen)
2 12.0107   # C (Carbon)
3 1.00794   # H (Hydrogen)
4 18.9984032 # F (Fluorine)
5 131.293   # Xe (Xenon)
6 14.0067   # N (Nitrogen)
```

#### Box Dimensions
```
0.0 23.86466077 xlo xhi
0.0 23.86466077 ylo yhi  
0.0 23.86466077 zlo zhi
```

## Computational Infrastructure

### LUMI Supercomputer Setup ([`lammps-gpu.sh`](./lammps-gpu.sh))

#### Hardware Configuration
```bash
#SBATCH --partition=standard-g    # GPU partition
#SBATCH --nodes=1                 # Single node
#SBATCH --ntasks-per-node=8       # 8 MPI ranks
#SBATCH --gpus-per-node=8         # 8 AMD MI250X GPUs
#SBATCH --time=48:00:00          # 48-hour time limit
#SBATCH --exclusive              # Exclusive node access
```

#### Software Environment
```bash
module load LUMI/24.03
module load LAMMPS/stable-2Aug2023-update4-pair-allegro-rocm-6.0.3-pytorch-2.3.1-20241203
```

#### Multi-GPU Execution
```bash
# Kokkos GPU acceleration
srun -n 8 lmp -sf kk -k on g 8 -pk kokkos gpu/aware off newton on neigh full
```

### Performance Optimization

#### GPU-Aware MPI
```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

#### Kokkos Configuration
- **GPU acceleration**: 8 AMD MI250X GPUs
- **Neighbor lists**: Full neighbor lists
- **GPU-aware MPI**: Disabled for stability

## Model Integration

### Allegro MLIP-2
- **File**: `2xe_cc3_tba_new-deployed_vf.pth` (to be found in the [Zenodo](https://doi.org/10.5281/zenodo.17105321) repository as `3-2xe_cc3_tba_new-deployed_vf.pth.tar.bz2`)

### Pair Style Configuration
```lammps
pair_style allegro
pair_coeff * * 2xe_cc3_tba_new-deployed_vf.pth O C H F Xe N
```

## Output Files

### Trajectory Files
- **equilibration.dump**: Equilibration phase trajectory (1 000 frames)
- **production.dump**: Production run trajectory (11 000 frames)
- **Format**: LAMMPS custom dump with positions, velocities, and forces

### Restart Files
- **equil.restart**: Equilibration checkpoints (every 10 000 steps)
- **production.restart**: Production checkpoints (every 10 000 steps)

### Log Files
- **test_lammps-gpu_*.out**: SLURM stdout with thermodynamic data
- **test_lammps-gpu_*.err**: SLURM stderr with error messages

## Usage Instructions

### 1. Prepare Input Files
```bash
# Ensure all input files are present:
ls 2xe_cc3_tba_lammps.in
ls 2xe_cc3_tba_str.data
ls 2xe_cc3_tba_new-deployed_vf.pth
```

### 2. Submit Job to LUMI
```bash
sbatch lammps-gpu.sh
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f test_lammps-gpu_*.out
```

## Requirements

### Computational Resources
- **HPC System**: [LUMI](https://docs.lumi-supercomputer.eu/) or equivalent GPU supercomputer
- **GPUs**: 8× AMD MI250X or NVIDIA A100 equivalent
- **Memory**: 64 GB system RAM, 16 GB GPU memory per rank
- **Storage**: 100 GB for trajectory files

### Software Dependencies
- **LAMMPS**: Version with Allegro pair style support
- **ROCm**: 6.0.3 or compatible GPU runtime
- **PyTorch**: 2.3.1 or compatible version
- **MPI**: GPU-aware implementation

---

For further details, please refer to the respective folders or contact the author via the provided email.
