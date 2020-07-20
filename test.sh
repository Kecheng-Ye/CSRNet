#!/bin/bash

num_model=$1

for ((i = 0 ; i < num_model ; i++)); do
    str="net_${i}.tar"
    python test.py --dataset_path data/part_B_final/test_data --model_name "${str}" --number_of_epoch "${i}"
    echo "Finish model ${i}"
done