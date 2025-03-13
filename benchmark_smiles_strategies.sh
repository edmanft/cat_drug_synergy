#!/bin/bash

# Number of seeds
num_seeds=10

# Define paths
data_path="data"
benchmark_dir="benchmark_smiles_strategies"
mkdir -p "$benchmark_dir"

# Loop over seeds
for ((seed=0; seed<num_seeds; seed++)); do

    echo "=== Running for seed = $seed ==="

    # --- OneHotEncoder with Morgan fingerprint ---
    for i in 512 1024 2048; do
        echo "Running OneHotEncoder with fpSize=$i, fill_strategy=zeros, seed=$seed"
        python src/scripts/train_classical_ml.py \
            --data_path "$data_path" \
            --encoder OneHotEncoder \
            --morgan_fp \
            --fpSize $i \
            --fill_strategy zeros \
            --seed $seed \
            --save_path "${benchmark_dir}/OneHotEncoder_morgan_fp_zeros_${i}_seed_${seed}.csv"

        echo "Running OneHotEncoder with fpSize=$i, fill_strategy=nan, seed=$seed"
        python src/scripts/train_classical_ml.py \
            --data_path "$data_path" \
            --encoder OneHotEncoder \
            --morgan_fp \
            --fpSize $i \
            --fill_strategy nan \
            --seed $seed \
            --save_path "${benchmark_dir}/OneHotEncoder_morgan_fp_nan_${i}_seed_${seed}.csv"
    done

    # --- OneHotEncoder without fingerprint ---
    echo "Running OneHotEncoder without Morgan fingerprint, seed=$seed"
    python src/scripts/train_classical_ml.py \
        --data_path "$data_path" \
        --encoder OneHotEncoder \
        --seed $seed \
        --save_path "${benchmark_dir}/OneHotEncoder_no_fp_seed_${seed}.csv"

done
