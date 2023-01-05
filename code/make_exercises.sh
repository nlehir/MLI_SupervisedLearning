#!/bin/bash
# array=($(ls))
# array=("k_means" "hierarchical_clustering" "metrics" "spectral_clustering", "pca/custom_data")
array=("linear_regression/1D_linear_regression" "linear_regression/dD_linear_regression")
for folder in "${array[@]}"
do
    printf "\n--\n"
    printf $folder
    printf "\n--\n"
    mkdir exercises\/$folder
    files=($(ls $folder | grep \\.py))
    for file in "${files[@]}"
    do
        echo $file
        diff -u $folder\/$file solutions\/$folder\/$file > exercises\/$folder\/$file.txt
    done
done
