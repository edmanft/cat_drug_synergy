#!/bin/bash

# Define the number of seeds
num_seeds=10  # Set this to the desired number of seeds

# Define paths
data_path="data"
save_dir="figures/ms/tables"
model_dir="models_weights"
benchmark_dir="benchmarks_embed_seeds"

# Create benchmark directory if it doesn't exist
mkdir -p "$benchmark_dir"

# Run benchmarks with the specified number of seeds
for ((seed=0; seed<num_seeds; seed++)); do
    model_dir_seed="${model_dir}/seed_${seed}"
    # PyTorch Tabular
    python src/scripts/train_pytorch_tabular.py --data_path "$data_path" --batch_size 512 --max_epoch 200 --es_patience 5  --seed "$seed" --save_path "$benchmark_dir/pytorch_tabular_seed_${seed}.csv" --model_dir $model_dir_seed 
   
    # OneHotEncoder
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder OneHotEncoder --seed 42 --save_path "$benchmark_dir/sklearn_OneHotEncoder_seed_${seed}.csv"

    # LabelEncoder
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder LabelEncoder --seed 42 --save_path "$benchmark_dir/sklearn_LabelEncoder_seed_${seed}.csv"

    # TabTransformer
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/TabTransformer_model.ckpt" --seed 42 --save_path "$benchmark_dir/sklearn_EmbeddingTabTransformer_seed_${seed}.csv"

    # AutoInt
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/AutoInt_model.ckpt" --seed 42 --save_path "$benchmark_dir/sklearn_EmbeddingAutoInt_seed_${seed}.csv"

    # CategoryEmbedding
    python src/scripts/train_classical_ml.py --data_path "$data_path" --encoder EmbeddingEncoder --model_path "$model_dir_seed/CategoryEmbedding_model.ckpt" --seed 42 --save_path "$benchmark_dir/sklearn_EmbeddingCategoryEmbedding_seed_${seed}.csv"

    
done
