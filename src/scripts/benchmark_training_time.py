#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Evaluating PyTorch Tabular Models.

This script loads the dataset, splits it into training, testing, and leaderboard sets,
and then iterates over a list of PyTorch Tabular models to train and evaluate them using
a consistent pipeline. The evaluation results are collected and displayed in a sorted DataFrame.

Usage:
    python train_pytorch_tabular_models.py --data_path path/to/data/ --verbose

Ensure that the dataset paths are correctly specified before running the script.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.exceptions import ConvergenceWarning
import logging
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pytorch_lightning as pl

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.models.category_embedding.config import CategoryEmbeddingModelConfig
from pytorch_tabular.models.autoint.config import AutoIntConfig
from pytorch_tabular.models.tabnet.config import TabNetModelConfig

import torch

from src.data.process_data import load_dataset, split_dataset
from src.model.evaluation import weighted_pearson, train_evaluate_pytorch_tabular_pipeline

# Suppress common warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def main():
    """
    Main function to execute the training and evaluation of PyTorch Tabular models.

    Example:
    python src/scripts/benchmark_training_time.py --data_path data --batch_size 512 --max_epoch 200 --es_patience 5
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating PyTorch Tabular Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument('--es_patience', type=int, default=30, help='Early stopping patience')
    args = parser.parse_args()

    


    # Suppress convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Define the file paths to your datasets
    drug_syn_path = os.path.join(args.data_path, 'drug_synergy.csv')
    cell_lines_path = os.path.join(args.data_path, 'cell_lines.csv')
    drug_portfolio_path = os.path.join(args.data_path, 'drug_portfolio.csv')

    # Check if files exist
    if not os.path.exists(drug_syn_path):
        print(f"Drug synergy file not found at {drug_syn_path}")
        exit(1)
    if not os.path.exists(cell_lines_path):
        print(f"Cell lines file not found at {cell_lines_path}")
        exit(1)
    if not os.path.exists(drug_portfolio_path):
        print(f"Drug portfolio file not found at {drug_portfolio_path}")
        exit(1)

    # Load and process the dataset
    full_dataset_df, column_type_dict = load_dataset(drug_syn_path, cell_lines_path, drug_portfolio_path)

    # Split the dataset into training, testing, and leaderboard sets
    datasets = split_dataset(full_dataset_df)

   
    # Use the standard embedding configuration
    categorical_cols = column_type_dict['categorical']['col_names']
    continuous_cols = column_type_dict['numerical']['col_names']


    # Define the list of PyTorch Tabular models to evaluate
    model_configs = [
        #('TabNet', TabNetModelConfig),
        ('CategoryEmbedding', CategoryEmbeddingModelConfig),
        #('Node', NodeConfig),
        ('AutoInt', AutoIntConfig),
        #('FTTransformer', FTTransformerConfig),
        ('TabTransformer', TabTransformerConfig),
    ]

    training_time_mean = []
    training_time_std = []

    # Iterate over each model, train and evaluate
    for model_name, ModelConfigClass in model_configs:
        print(f"Training and evaluating {model_name}...")
        training_time_list = []
        try:
            # Define configurations for PyTorch Tabular
            data_config = DataConfig(
                target=['synergy_score'],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
               )

            # Model-specific configurations
            model_config = ModelConfigClass(
                task="regression",
                metrics=["mean_squared_error"],
                metrics_params=[{}],
            )

            trainer_config = TrainerConfig(
                max_epochs=args.max_epoch,
                batch_size=args.batch_size,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use accelerator for device selection
                devices=1 if torch.cuda.is_available() else None,  # Number of devices (GPUs/CPUs)
                early_stopping="valid_loss",
                early_stopping_patience=args.es_patience,
            
            )

            for i in range(10):
                _, _, training_time = train_evaluate_pytorch_tabular_pipeline(
                    datasets=datasets,
                    data_config=data_config,
                    model_config=model_config,
                    trainer_config=trainer_config,
                )
                training_time_list.append(training_time)
            training_time_mean.append(np.mean(training_time_list))
            training_time_std.append(np.std(training_time_list))
            
            


        except Exception as e:
            print(f"Error training {model_name}: {e}\n")

    # Obtain mean and std of training time
    print("\nTraining Time Results:")
    training_time_df = pd.DataFrame({
        'Model': [model_name for model_name, _ in model_configs],
        'Mean Training Time (s)': training_time_mean,
        'Std Training Time (s)': training_time_std,
    })
    print(training_time_df)

if __name__ == '__main__':
    main()