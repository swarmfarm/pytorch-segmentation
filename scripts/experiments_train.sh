#!/bin/bash

# Define experiments as dataset:epochs pairs; the null values are placeholders for future experiment parameters
experiments=(
    # "09:100:null:null"
    # "09:200:null:null"
    # "09:500:null:null"
    # "9:1000:null:null"
    # "11:100:null:null"
    # "11:200:null:null"
    "11:500:null:null"
    # "11:1000:null:null"
)

# Iterate over the experiments
for experiment in "${experiments[@]}"; do
    # Split the experiment into dataset and epochs  (with placeholders for more experiment parameters)
    IFS=':' read -r dataset epoch unused1 unused2 <<< "$experiment"
    
    echo "Running experiment: dataset=$dataset, epochs=$epoch"
    python3 train.py /home/nvidia/Downloads/keymakr/batch_$(printf "%02d" $dataset)/ \
        --dataset=keymakr \
        --map-classes \
        --epochs=$epoch \
        --model-dir=./results/keymakr_batch_$(printf "%02d" $dataset)_epochs_$(printf "%04d" $epoch)_plus-obstacles/
done
