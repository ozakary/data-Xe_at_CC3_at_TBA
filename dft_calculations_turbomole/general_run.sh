# define step
# for d in */; do cp define_input.sh "$d"; done
# for d in ./*/ ; do (cd "$d" && define < define_input.sh ); done

 for d in ./*/ ; do (cd "$d" && actual -r ); done

# send calculations
 for d in */; do cp -f tm78_puhti.job "$d"; done
for d in ./*/ ; do (cd "$d" && sbatch tm78_puhti.job ); done


