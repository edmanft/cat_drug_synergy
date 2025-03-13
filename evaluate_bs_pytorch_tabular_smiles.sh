#!/bin/bash

# Fixed seed
seed=0

# Define paths
data_path="data"
benchmark_dir="train_pytorch_tabular_different_fingerprints"
mkdir -p "$benchmark_dir"

# Loop over fingerprint sizes
for fpSize in 1024 2048; do
    # Loop over batch sizes
    for batch_size in 32 64 128 256 512; do
        echo "Running PyTorch Tabular with fpSize=$fpSize, batch_size=$batch_size, seed=$seed"

        python src/scripts/train_pytorch_tabular.py \
            --data_path "$data_path" \
            --morgan_fp \
            --fpSize $fpSize \
            --fill_strategy nan \
            --batch_size $batch_size \
            --max_epoch 200 \
            --es_patience 5 \
            --seed $seed \
            --save_path "${benchmark_dir}/pytorch_tabular_fp_${fpSize}_batch_${batch_size}_seed_${seed}.csv"
    done
done



# Loop over batch sizes
for batch_size in 32 64 128 256 512; do
    echo "Running PyTorch Tabular without fingerprints, batch_size=$batch_size, seed=$seed"

    python src/scripts/train_pytorch_tabular.py \
        --data_path "$data_path" \
        --batch_size $batch_size \
        --max_epoch 200 \
        --es_patience 5 \
        --seed $seed \
        --save_path "${benchmark_dir}/pytorch_tabular_no_fp_batch_${batch_size}_seed_${seed}.csv"
done