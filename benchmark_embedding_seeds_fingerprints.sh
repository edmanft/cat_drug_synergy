#!/bin/bash

# Define the number of seeds
num_seeds=10  # Set this to the desired number of seeds

# Define paths
data_path="data"
model_dir="models_weights_fingerprints_1024"
benchmark_dir="benchmarks_embed_seeds_fingerprints_1024"

# Create benchmark directory if it doesn't exist
mkdir -p "$benchmark_dir"

# Run benchmarks with the specified number of seeds
for ((seed=0; seed<num_seeds; seed++)); do
    model_dir_seed="${model_dir}/seed_${seed}"
    # PyTorch Tabular
    echo "==> Running PyTorch Tabular for seed $seed"
    python src/scripts/train_pytorch_tabular.py --data_path "$data_path" --batch_size 32 --max_epoch 200 --es_patience 5 --morgan_fp --fpSize 1024 --fill_strategy nan   --seed "$seed" --save_path "$benchmark_dir/pytorch_tabular_seed_${seed}.csv" --model_dir $model_dir_seed 
   
    # OneHotEncoder
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder OneHotEncoder --seed $seed  --morgan_fp --fpSize 1024 --fill_strategy nan --save_path "$benchmark_dir/sklearn_OneHotEncoder_seed_${seed}.csv"

    # LabelEncoder
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder LabelEncoder  --morgan_fp --fpSize 1024 --fill_strategy nan --seed $seed --save_path "$benchmark_dir/sklearn_LabelEncoder_seed_${seed}.csv"

    # TabTransformer
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/TabTransformer_model.ckpt" --seed $seed  --morgan_fp --fpSize 1024 --fill_strategy nan --save_path "$benchmark_dir/sklearn_EmbeddingTabTransformer_seed_${seed}.csv"

    # CategoryEmbedding
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/CategoryEmbedding_model.ckpt" --seed $seed  --morgan_fp --fpSize 1024 --fill_strategy nan --save_path "$benchmark_dir/sklearn_EmbeddingCategoryEmbedding_seed_${seed}.csv"

    
done
