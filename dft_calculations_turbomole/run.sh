#!/bin/bash

# carpo2
# module load turbomole2024

# for i in "30509" ; do
# for i in $(seq -f "%04g" 1509 500 9509); do
 for i in $(seq -f "%01g" 60801 100 69901 ); do
    cp -f  lumi_tm78.job ./cluster_${i}
    cd ./cluster_${i}
#     define < define_input
     sbatch  lumi_tm78.job 
     cd ..
done



