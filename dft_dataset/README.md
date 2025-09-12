# Dataset Formatting for Allegro Architecture

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

This directory contains the [code](./extract_vasp_to_xyz.py) for converting VASP calculation output file (OUTCAR) into the .xyz format required by the Allegro architecture.

## Input Data

### Source Files
- **OUTCAR files**: Located in `vasp_folders_2Xe_CC3_TBA/DFT-NMR_iter-{i}/OUT-{i}/OUTCAR`
- **POSCAR files**: Located in `vasp_folders_2Xe_CC3_TBA/DFT-NMR_iter-{i}/OUT-{i}/POSCAR`
- **Configuration range**: i = 0 to 13740 (increment of 60)
- **Total configurations**: 230 snapshots

### Extracted Properties
From OUTCAR files:
- **Atomic coordinates** (Cartesian, Å)
- **Atomic forces** (eV/Å)
- **Total energy** (eV)
- **Stress tensor** (eV/Å<sup>3</sup>)
- **Unit cell volume** (Å<sup>3</sup>)

From POSCAR files:
- **Atomic species** (element symbols)
- **Lattice parameters** (Å)
- **Number of atoms per species**

## Output Format

### .xyz File Structure
The code generates .xyz files in the extended format required by Allegro:

```
1170
Lattice="23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077" Properties=species:S:1:pos:R:3:forces:R:3 original_dataset_index=0 energy=-7265.49063459 stress="-0.022487536484425603 -0.00036718892857503117 -0.00040346614670645043 -0.00036718892857503117 -0.022938923934245986 3.514930316375881e-05 -0.00040346614670645043 3.514930316375881e-05 -0.022958062436619624" free_energy=-7265.49063459 pbc="T T T"
O       23.44491000     19.94573000     16.86242000     -0.72071600     -1.17918600     0.93678100
C       23.89142000     15.92919000     15.65737000     0.44649600      3.71207100      2.15138200
...
```

### Header Line Components
- **Line 1**: Number of atoms (1170)
- **Line 2**: Metadata including:
  - `Lattice`: 3×3 lattice vectors (flattened)
  - `Properties`: Data format specification
  - `original_dataset_index`: Configuration identifier
  - `energy`: Total DFT energy (eV)
  - `stress`: 3×3 stress tensor (GPa, flattened)
  - `free_energy`: Same as energy
  - `pbc`: Periodic boundary conditions (T T T)

### Data Lines
Each subsequent line contains:
- **Element symbol** (O, C, H, F, Xe, N)
- **x, y, z coordinates** (Å, 8 decimal places)
- **Fx, Fy, Fz forces** (eV/Å, 8 decimal places)

## Code Description

### Main Functions

#### `extract_coordinates_and_forces(outcar_file, poscar_file)`
- Parses OUTCAR for atomic positions and forces
- Extracts atomic species from POSCAR
- Returns numpy arrays of coordinates, forces, and species list

#### `write_coordinates_and_forces(...)`
- Formats data according to Allegro specifications
- Writes individual .xyz files for each configuration
- Maintains 8 significant digits precision

#### `concatenate_ML_files(file_paths, output_file)`
- Combines individual .xyz files into a single trajectory
- Creates `ML_total` file containing all configurations

#### `modify_dataset_indexes(input_file, output_file, start_index)`
- Renumbers `original_dataset_index` values
- Creates `ML_total_modified` with sequential indexing starting from 197

## Usage

### Running the Conversion Script
```bash
python extract_vasp_to_xyz.py
```

### Output Files Generated
- `ML_files/ML_{i}.txt`: Individual .xyz files for each configuration
- `ML_total`: Concatenated trajectory file
- `ML_total_modified`: Final dataset with reindexed configurations

### Alternative Method (ASE)
For simpler conversions, ASE can be used:
```bash
ase convert OUTCAR filename.xyz
```
## Dependencies

```python
import os
import numpy as np
import re
```

### System Requirements
- **Python 3.x** with NumPy
- **VASP output files** (OUTCAR, POSCAR)
- **Sufficient disk space** for .xyz files

---

For further details, please refer to the respective folders or contact the author via the provided email.	
