# Semi-Empirical Molecular Dynamics (SEMD) for Xe@CC3@TBA

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)
---

This directory contains the input file for running semi-empirical molecular dynamics (SEMD) with DFTB+ code for Xe@CC3@TBA.

## System Description
CC3 POCs in 4-(trifluoromethoxy)benzyl alcohol (TBA) solvent loaded with high Xe concentration:
- **1170 atoms**
- **Atom types**: H, C, N, O, F, Xe
- **Box size (PBC)**: 23.865 Å × 23.865 Å × 23.865 Å cubic cell

## Simulation Parameters

### Method
- **Hamiltonian**: GFN2-xTB semi-empirical method
- **Electronic structure**: 
  - Fermi-Dirac distribution at 300 K
  - Γ-point only calculation (1×1×1 k-point grid)

### Molecular Dynamics Settings
- **Integrator**: Velocity-Verlet algorithm
- **Time step**: 0.5 fs
- **Total steps**: 2000 (1.0 ps total simulation time)
- **Temperature**: 300 K (maintained via Nosé-Hoover thermostat)
- **Thermostat parameters**:
  - Chain length: 3
  - Coupling strength: 5 × 10<sup>12</sup> Hz
- **Restart frequency**: Every 10 steps

### Initial Conditions
- **Velocities**: Maxwell-Boltzmann distribution at 300 K (provided in input)
- **Starting geometry**: Optimized host-guest configuration

## Input Files

- `dftb_in.hsd`: Main DFTB+ input file containing geometry, MD parameters, and method settings
- `geometry.gen`: Initial atomic coordinates (included in main input file)
- `detailed.out`: Detailed output from DFTB+ simulation
- `md.out`: MD trajectory file
- `results.tag`: Results summary file

## Files in This Directory

- [`./dftb_in.hsd`](./dftb_in.hsd): Main DFTB+ input file containing geometry, MD parameters, and method settings

## Dependencies

- **DFTB+**: Version compatible with GFN2-xTB parameters
---

For further details, please refer to the respective folders or contact the author via the provided email.
