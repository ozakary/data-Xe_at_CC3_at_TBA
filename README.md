# Supporting Data for “Equivariant Neural Networks Reveal How Host–Guest Interactions Shape <sup>129</sup>Xe NMR in Porous Liquids”

## Graphical Abstract

![Graphical Abstract](./blank.png)

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)
---

This is the Supporting Code for the manuscript “Equivariant Neural Networks Reveal How Host–Guest Interactions Shape <sup>129</sup>Xe NMR in Porous Liquids”. [DOI: TBA]

The repository contains the following sections:

1. Code for dataset preparation for MLIP:  
   i. Configuration generation using semi-empirical MD (SEMD) simulations. ([directory](./dftb-md/))  
   ii. Single-point DFT calculations of the generated SEMD configurations. ([directory](./dft_calculations_vasp/))  
   iii. Dataset formatting for *Allegro* architecture. ([directory](./dft_dataset/))  
3. Code for training, validation, and testing of *Allegro* architecture. ([directory](./allegro_achitecture/))  
4. Code for machine learning MD simulations. ([directory](./mlmd_simulations/))  
5. Code for dataset preparation for the NMR-ML model:  
   i. Configuration generation using SEMD simulations. ([directory](./dftb-md/))  
   ii. DFT calculations of <sup>129</sup>Xe NMR magnetic shielding tensor, ***σ***. ([directory](./dft_calculations_turbomole/))  
   iii. Dataset formatting for *MatTen* architecture. ([directory](./nmr-ml_dataset/))  
6. Code for training, validation, and testing of *MatTen*. ([directory](./matten_architecture/))  
7. Prediction of <sup>129</sup>Xe ***σ*** from the pre-trained NMR-ML model. ([directory](./nmr-ml_prediction/))  
8. Python scripts and raw numerical data for all figures in the main manuscript and the supporting information. ([directory](./figures/))  

## Citations

If you use this data, please cite the following: \
\
**Paper:** Zakary, O.; Lantto P. Equivariant Neural Networks Reveal How Host–Guest Interactions Shape <sup>129</sup>Xe NMR in Porous Liquids. *In preparation* **2025**. [DOI: TBA]

\
**Dataset:** Zakary, O.; Lantto P. (**2025**). Supporting Data for “Equivariant Neural Networks Reveal How Host–Guest Interactions Shape <sup>129</sup>Xe NMR in Porous Liquids”. *figshare. Dataset.* [DOI: TBA]

---

For further details, please refer to the respective folders or contact the author via the provided email.
