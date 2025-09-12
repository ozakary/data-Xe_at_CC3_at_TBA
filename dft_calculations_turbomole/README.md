# DFT Calculations of <sup>129</sup>Xe NMR Magnetic Shielding Tensor

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the input files and job scripts for calculating <sup>129</sup>Xe NMR magnetic shielding tensors using TURBOMOLE at three different levels of theory.

## Computational Levels of Theory

### 1. BHandHLYP/SVP
- **Functional**: BHandHLYP
- **Basis set**: SVP for other atoms and TZVP for Xe
- **Directory**: `./BHandHLYP_SVP/`

### 2. PBE/SVP  
- **Functional**: PBE
- **Basis set**: SVP for other atoms and TZVP for Xe
- **Directory**: `./PBE_SVP/`

### 3. PBE/TZVP
- **Functional**: PBE
- **Basis set**: TZVP
- **Directory**: `./PBE_TZVP/`

## TURBOMOLE Configuration

### Input Generation ([`define_input`](./define_input))

#### Relativistic Treatment
```
x2c                    # Exact two-component relativistic method
rlocal y               # Local relativistic corrections
finnuc y               # Finite nucleus model
```

#### Basis Set Specification
```
b all x2c-SVPall                    # SVP basis for all atoms
b "xe" x2c-TZVPall-s               # Enhanced TZVP basis for Xe (PBE/TZVP only)
```

#### DFT Settings
```
dft
on
func pbe               # PBE functional (or bhandhlyp)
grid 4a                # High-quality integration grid
dsp
d4                     # DFT-D4 dispersion correction
```

#### NMR-Specific Parameters
```
nmr
nucsel "xe"            # Select xenon nuclei for NMR calculation
shiftconv 5            # Convergence threshold for chemical shifts
csmaxiter 70           # Maximum iterations for chemical shift calculation
```

#### Performance Optimization
```
ri
on                     # Resolution of identity approximation
m 1000                 # Memory allocation (MB)
marij                  # RI for exchange-correlation
mp2
memory 3500            # MP2 memory (MB, if needed)
```

## Computational Workflow

### Directory Structure
```
./[LEVEL_OF_THEORY]/
├── lumi_tm78.job              # SLURM job script
├── run.sh                     # Batch job submission
├── prepare.sh                 # Input file preparation
├── general_run.sh             # Workflow management
├── define_input              # TURBOMOLE input template
├── COORDS/                   # Source coordinate files
│   └── 2xe_tba_cc3_coords.*.xyz
└── FINISHED/                 # Completed calculations
    └── cluster_*/            # Individual calculation directories
```

### Preparation Workflow ([`prepare.sh`](./prepare.sh))

#### Coordinate Processing
1. **Source extraction**: Extract configurations from trajectory files
2. **Cluster generation**: Create local environments around Xe nuclei
3. **Format conversion**: Convert XYZ to TURBOMOLE coordinate format
4. **Environment selection**: Select atoms within interaction radius

#### Clustering Script
```python
# Extract local environment around Xe nuclei
Xe2CC3TBA_getNMolsAroundCentralOne3D.py coord_$i.xyz 20
```

### Job Execution ([`lumi_tm78.job`](./lumi_tm78.job))

#### SLURM Configuration
```bash
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # 64 CPU cores
#SBATCH --time=04:00:00           # 4-hour time limit
#SBATCH --mem-per-cpu=1750        # 1.75 GB memory per core
#SBATCH --account=project_462000738
```

#### TURBOMOLE Environment
```bash
export TURBODIR=/projappl/project_462000738/TURBOMOLE_7.8/TURBOMOLE
export PARA_ARCH=SMP             # Shared memory parallelization
export OMP_NUM_THREADS=64        # OpenMP threading
export PARNODES=64               # Parallel nodes
```

#### Calculation Sequence
```bash
define < define_input            # Generate input files
ridft > ridft.out               # SCF calculation
mpshift > mpshift.out           # NMR shielding calculation
```

## Batch Processing System

### Configuration Range
- **Frame indices**: 201 to 99,901 (increment: 100)
- **Total configurations**: ~999 snapshots per level of theory
- **Parallel execution**: Multiple SLURM jobs simultaneously

### Workflow Management ([`general_run.sh`](./general_run.sh))

```bash
# Step 1: Generate TURBOMOLE input files
for d in ./*/ ; do (cd "$d" && define < define_input ); done

# Step 2: Activate calculations  
for d in ./*/ ; do (cd "$d" && actual -r ); done

# Step 3: Distribute job scripts
for d in */; do cp -f tm78_puhti.job "$d"; done

# Step 4: Submit all calculations
for d in ./*/ ; do (cd "$d" && sbatch tm78_puhti.job ); done
```

#### Workflow Steps Explained
1. **Input generation**: Runs `define` in each calculation directory to create TURBOMOLE input files
2. **Calculation activation**: Uses `actual -r` to prepare the calculation environment
3. **Job script distribution**: Copies the SLURM job script to each directory
4. **Batch submission**: Submits all jobs to the queue system

### Individual Job Submission ([`run.sh`](./run.sh))
```bash
for i in $(seq -f "%01g" 60801 100 69901); do
    cp -f lumi_tm78.job ./cluster_${i}
    cd ./cluster_${i}
    sbatch lumi_tm78.job
    cd ..
done
```

## Output Analysis

### Key Output Files (per calculation)
- **ridft.out**: SCF convergence and electronic structure
- **mpshift.out**: NMR shielding tensor components
- **coord**: TURBOMOLE coordinate file
- **slurm-*.out**: Job execution log

## Data Processing Pipeline

### 1. Coordinate Preparation
```bash
# Extract configurations from MD trajectory
python extract_coordinates.py trajectory.xyz

# Create local clusters around Xe
python Xe2CC3TBA_getNMolsAroundCentralOne3D.py
```

### 2. TURBOMOLE Input Generation
```bash
# Prepare calculation directories
./prepare.sh

# Submit batch calculations
./run.sh
```

### 3. Results Collection
```bash
# Extract shielding values from output files
python extract_nmr_data.py ./FINISHED/cluster_*/mpshift.out
```
## Computational Requirements

### Hardware Specifications
- **Supercomputer**: [LUMI](https://docs.lumi-supercomputer.eu/)
- **CPU**: 64 cores per calculation
- **Memory**: 112 GB per job (1.75 GB × 64 cores)
- **Storage**: ~100 MB per calculation
- **Time**: 2-4 hours per configuration

### Software Dependencies
- **TURBOMOLE**: Version 7.8 with NMR module
- **Python**: Coordinate processing scripts

---

For further details, please refer to the respective folders or contact the author via the provided email.	
