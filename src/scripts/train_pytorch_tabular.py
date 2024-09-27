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
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating PyTorch Tabular Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--no_embedding', action='store_true', help='Use OneHotEncoding instead of embeddings for categorical variables.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output.')
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

    # Define categorical and continuous features based on the `--no_embedding` flag
    if args.no_embedding:
        # If no embedding is specified, apply OneHotEncoding to categorical columns
        print("Applying OneHotEncoding to categorical variables...")

        categorical_cols = column_type_dict['categorical']['col_names']
        continuous_cols = column_type_dict['numerical']['col_names']

        # Perform OneHotEncoding on the categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
            ]
        )
        preprocessor.set_output(transform='pandas')
        datasets['train']['X'] = preprocessor.fit_transform(datasets['train']['X'])

        for split in ['test', 'lb']:
            datasets[split]['X'] = preprocessor.transform(datasets[split]['X'])

        # Since we've OneHotEncoded, categorical columns will be empty
        categorical_cols = []
        continuous_cols = list(datasets['train']['X'].columns)


    else:
        # Use the standard embedding configuration
        categorical_cols = column_type_dict['categorical']['col_names']
        continuous_cols = column_type_dict['numerical']['col_names']


    # Define the list of PyTorch Tabular models to evaluate
    model_configs = [
        ('TabNet', TabNetModelConfig),
        #('CategoryEmbedding', CategoryEmbeddingModelConfig),
        #('Node', NodeConfig),
        #('AutoInt', AutoIntConfig),
        #('FTTransformer', FTTransformerConfig),
        #('TabTransformer', TabTransformerConfig),
    ]

    # Dictionary to store evaluation results
    evaluation_results: Dict[str, Dict[str, Any]] = {}

    # Iterate over each model, train and evaluate
    for model_name, ModelConfigClass in model_configs:
        print(f"Training and evaluating {model_name}...")
        try:
            # Define configurations for PyTorch Tabular
            data_config = DataConfig(
                target=['synergy_score'],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
               )

            # Model-specific configurations
            model_config = ModelConfigClass(
                seed=42,
                task="regression",
                metrics=["mean_squared_error"],
                metrics_params=[{}],
            )

            trainer_config = TrainerConfig(
                max_epochs=5,
                batch_size=512,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use accelerator for device selection
                devices=1 if torch.cuda.is_available() else None,  # Number of devices (GPUs/CPUs)
                early_stopping="valid_loss",
                early_stopping_patience=30,
            )

            # Call the training and evaluation pipeline
            eval_dict = train_evaluate_pytorch_tabular_pipeline(
                datasets=datasets,
                data_config=data_config,
                model_config=model_config,
                trainer_config=trainer_config,
                verbose=args.verbose
            )

            evaluation_results[model_name] = eval_dict
            print(f"{model_name} - Train Weighted Pearson Correlation: {eval_dict['train']['wpc']:.4f}")
            print(f"{model_name} - Test Weighted Pearson Correlation: {eval_dict['test']['wpc']:.4f}")
            print(f"{model_name} - LB Weighted Pearson Correlation: {eval_dict['lb']['wpc']:.4f}\n")

        except Exception as e:
            print(f"Error training {model_name}: {e}\n")

    # Collect results into lists for DataFrame creation
    model_list = []
    train_wpc_list = []
    test_wpc_list = []
    lb_wpc_list = []

    for name, eval_dict in evaluation_results.items():
        model_list.append(name)
        train_wpc_list.append(eval_dict['train']['wpc'])
        test_wpc_list.append(eval_dict['test']['wpc'])
        lb_wpc_list.append(eval_dict['lb']['wpc'])

    # Create a DataFrame to display evaluation results
    evaluation_df = pd.DataFrame({
        'Model': model_list,
        'Train WPC': train_wpc_list,
        'Test WPC': test_wpc_list,
        'LB WPC': lb_wpc_list
    })

    # Sort the DataFrame based on the Leaderboard Weighted Pearson Correlation
    evaluation_df = evaluation_df.sort_values(by='LB WPC', ascending=False)
    print("Evaluation Results:")
    print(evaluation_df.reset_index(drop=True))

if __name__ == '__main__':
    main()