#!/bin/bash

# Define the number of seeds
num_seeds=10  # Set this to the desired number of seeds

# Define paths
data_path="data"
model_dir="models_weights"
benchmark_dir="benchmarks_sklearn_embed_seeds"

# Create benchmark directory if it doesn't exist
mkdir -p "$benchmark_dir"

# Run benchmarks with the specified number of seeds
for ((seed=0; seed<num_seeds; seed++)); do
    model_dir_seed="${model_dir}/seed_${seed}"
   
   

    # TabTransformer
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/TabTransformer_model.ckpt" --seed $seed --save_path "$benchmark_dir/sklearn_EmbeddingTabTransformer_seed_${seed}.csv"

    # AutoInt
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/AutoInt_model.ckpt" --seed $seed --save_path "$benchmark_dir/sklearn_EmbeddingAutoInt_seed_${seed}.csv"

    # CategoryEmbedding
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/CategoryEmbedding_model.ckpt" --seed $seed --save_path "$benchmark_dir/sklearn_EmbeddingCategoryEmbedding_seed_${seed}.csv"

    
done
