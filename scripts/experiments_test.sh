#!/bin/bash

# Define experiments as dataset:epochs pairs; the null values are placeholders for future experiment parameters
experiments=(
    # "9:100:11:null"
    # "11:100:9:null"
    # "9:200:11:null"
    # "9:1000:11:null"
    "11:200:9:null"
    # "9:500:11:null"
    # "9:1000:11:null"
    # "11:200:9:null"
)

# Iterate over the experiments
for experiment in "${experiments[@]}"; do
    # Split the experiment into model, dataset and epochs  (with placeholders for more experiment parameters)
    IFS=':' read -r dataset epoch test_dataset unused2 <<< "$experiment"

    model="./results/keymakr_batch_$(printf "%02d" $dataset)_epochs_$(printf "%04d" $epoch)_plus-obstacles/model_best.pth"
    echo "Testing experiment: model=$model, dataset=$test_dataset"
    python3 train.py /home/nvidia/Downloads/keymakr/batch_$(printf "%02d" $test_dataset)/ \
        --dataset=keymakr \
        --map-classes \
        --test-only \
        --resume $model

done
