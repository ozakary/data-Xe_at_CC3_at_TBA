# Single-Point DFT Calculations

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)
---

This directory contains the input files and job script for performing single-point DFT calculations using VASP.

## Computational Parameters (INCAR)

- **Functional**: PBE with D4 dispersion correction
- **Plane-wave cutoff**: 1000 eV
- **Precision**: Accurate
- **Electronic convergence**: 1×10<sup>-6</sup> eV
- **Smearing**: Gaussian, 0.01 eV
- **SCF settings**: 
  - Maximum iterations: 100
  - Minimum iterations: 6
- **Real-space projection**: Automatic
- **Parallelization**: 64 bands per processor
- **Wave functions**: Not written
- **Charge density**: Not written
- **k-point grid**: 1×1×1 (Γ-point only)

## Pseudopotentials (POTCAR)
- **O**: PAW_PBE O_GW_new (6 valence electrons)
- **C**: PAW_PBE C_GW_new (4 valence electrons)  
- **H**: PAW_PBE H_GW_new (1 valence electron)
- **F**: PAW_PBE F_GW_new (7 valence electrons)
- **N**: PAW_PBE N_GW_new (5 valence electrons)
- **Xe**: PAW_PBE Xe_sv_GW (26 valence electrons, includes 4d electrons)

## Directory Structure

```
./all_PBE_DFT.job
./vasp_folders_2Xe_CC3_TBA/
├── DFT-NMR_iter-0/
│   ├── IN/
│   │   ├── INCAR
│   │   ├── KPOINTS
│   │   ├── POTCAR
│   │   └── POSCAR
│   └── OUT-0/
├── DFT-NMR_iter-60/
│   ├── IN/
│   └── OUT-60/
...
└── DFT-NMR_iter-13740/
    ├── IN/
    └── OUT-13740/
```

### Batch Processing
The job script processes configurations in batches with increment of 60:
- Starting configuration: iter-0
- Ending configuration: iter-13740
- Total batches: 230 configurations

## Input Files

### Required Files (provided)
- [`INCAR`](./INCAR): DFT calculation parameters
- [`KPOINTS`](./KPOINTS): k-point sampling specification
- [`POTCAR`](./POTCAR): Pseudopotential data for all elements
- [`POSCAR`](./POSCAR): Atomic coordinates for each configuration

### Generated Files
- [`all_PBE_DFT.job`](./all_PBE_DFT.job): SLURM batch job script
- Individual [`POSCAR`](./POSCAR) files for each snapshot

## Output Files

The calculations generate standard VASP output files in each `OUT-{iter}` directory:
- `OUTCAR`: Detailed calculation results (energies, forces, stress)
- `OSZICAR`: SCF convergence information
- `vasprun.xml`: Complete calculation data in XML format
- `CONTCAR`: Final atomic positions (unchanged for single-point)

**Note**: Output files are available in the [Zenodo repository](https://github.com/ozakary/data-Xe_at_CC3_at_TBA).

## Requirements

- **Supercomputer**: Mahti supercomputer at CSC - IT Center for Science (Finland), more details: [www.csc.fi](https://www.csc.fi)
- **VASP**: Version with DFT-D4 support (vasp_std_dftd4)

---

For further details, please refer to the respective folders or contact the author via the provided email.
