#!/bin/bash

# Define the number of seeds
num_seeds=10  # Set this to the desired number of seeds

# Define paths
data_path="data"
save_dir="figures/ms/tables"
model_dir="models_weights/seed_0"
benchmark_dir="benchmarks_sklearn_seed_0"

# Create benchmark directory if it doesn't exist
mkdir -p "$benchmark_dir"

# Run benchmarks with the specified number of seeds
for ((seed=0; seed<num_seeds; seed++)); do
    # OneHotEncoder
    #python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder OneHotEncoder --seed "$seed" --save_path "$benchmark_dir/sklearn_OneHotEncoder_seed_${seed}.csv"

    # LabelEncoder
    #python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder LabelEncoder --seed "$seed" --save_path "$benchmark_dir/sklearn_LabelEncoder_seed_${seed}.csv"

    # TabTransformer
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir/TabTransformer_model.ckpt" --seed "$seed" --save_path "$benchmark_dir/sklearn_EmbeddingTabTransformer_seed_${seed}.csv"

    # AutoInt
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir/AutoInt_model.ckpt" --seed "$seed" --save_path "$benchmark_dir/sklearn_EmbeddingAutoInt_seed_${seed}.csv"

    # CategoryEmbedding
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir/CategoryEmbedding_model.ckpt" --seed "$seed" --save_path "$benchmark_dir/sklearn_EmbeddingCategoryEmbedding_seed_${seed}.csv"

done
