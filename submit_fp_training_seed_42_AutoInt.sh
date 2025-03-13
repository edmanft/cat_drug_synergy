#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-03:00:00
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=mlip_synergy
#SBATCH --output=logs/seed_%a.out
#SBATCH --error=logs/seed_%a.err

module load mamba
source activate cat_drug_synergy

seed=42

data_path="data"
model_dir="models_weights_fingerprints_1024/seed_${seed}"
benchmark_dir="benchmarks_embed_seeds_fingerprints_1024"
mkdir -p "$benchmark_dir"

echo "==> Running PyTorch Tabular for seed $seed"

# AutoInt
python src/scripts/train_classical_ml.py \
    --data_path "$data_path" \
    --encoder EmbeddingEncoder \
    --model_path "$model_dir/AutoInt_model.ckpt" \
    --morgan_fp --fpSize 1024 --fill_strategy nan \
    --seed $seed \
    --save_path "$benchmark_dir/sklearn_EmbeddingAutoInt_seed_${seed}.csv" &

wait

