#!/bin/bash

# carpo2
 module load turbomole
## module load python/3.7.4

# puhti
# module load turbomole
# module load python-env

#split --additional-suffix=.xyz -d -a 4 -l 1950 xtb.trj coord_

# for i in "30509" ; do
# for i in $(seq -f "%04g" 1509 500 9509); do
 for i in $(seq -f "%01g" 201 100 99901 ); do
    mkdir cluster_${i}
    cp -f ./COORDS/2xe_tba_cc3_coords.${i}.xyz  ./cluster_${i} 
    cd ./cluster_${i}
    head -2 2xe_tba_cc3_coords.${i}.xyz > coord_$i.xyz     
    grep -A 1170 "Lattice"  2xe_tba_cc3_coords.${i}.xyz | grep -v Lattice | awk '{print $1,$2,$3,$4}' >> coord_$i.xyz
## puhti
    /projappl/plantto/bin/pythonScritdir/Xe2CC3TBA_getNMolsAroundCentralOne3D.py coord_$i.xyz 20 
## carpo2
##    /home/plantto//bin/pythonScritdir/XeTBA_getNMolsAroundCentralOne3D.py coord_${i}.xyz 10 0 > new_coord_${i}_10.xyz
     x2t coord_${i}ClusterAroundXeNew.xyz > coord
     cp ../define_input .
     cp ../tm78_puhti.job .
#     define < define_input
#     sbatch tm78_puhti.job
     cd ..
done

# ./run_turbomole.sh


