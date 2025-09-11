grep -vE ',(0\.0|0)(,|$)' dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv > ms_verif.csv
(head -n 1 dataset_schnet_atomic_coordinates/structures.csv && grep ',Xe,' dataset_schnet_atomic_coordinates/structures.csv) > str_verif.csv
