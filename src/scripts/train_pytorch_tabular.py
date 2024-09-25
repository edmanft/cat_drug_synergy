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
from src.model.evaluation import weighted_pearson

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

    # Combine the datasets for PyTorch Tabular (it expects a single DataFrame)
    train_df = datasets['train']['X'].copy()
    train_df['target'] = datasets['train']['y'].values
    train_df['comb_id'] = datasets['train']['comb_id']

    test_df = datasets['test']['X'].copy()
    test_df['target'] = datasets['test']['y'].values
    test_df['comb_id'] = datasets['test']['comb_id']

    lb_df = datasets['lb']['X'].copy()
    lb_df['target'] = datasets['lb']['y'].values
    lb_df['comb_id'] = datasets['lb']['comb_id']

    # For PyTorch Tabular, we'll combine all data and specify splits
    full_df = pd.concat([train_df, test_df, lb_df], axis=0)
    full_df.reset_index(drop=True, inplace=True)

    # Create a 'split' column to indicate the dataset split
    num_train = len(train_df)
    num_test = len(test_df)
    full_df['split'] = 'lb'
    full_df.loc[:num_train-1, 'split'] = 'train'
    full_df.loc[num_train:num_train+num_test-1, 'split'] = 'test'

    # Define categorical and continuous features
    categorical_cols = column_type_dict['categorical']['col_names']
    continuous_cols = column_type_dict['numerical']['col_names']

    target_col = 'target'

    # Define the list of PyTorch Tabular models to evaluate
    model_configs = [
        ('TabNet', TabNetModelConfig),
        ('CategoryEmbedding', CategoryEmbeddingModelConfig),
        ('Node', NodeConfig),
        ('AutoInt', AutoIntConfig),
        ('FTTransformer', FTTransformerConfig),
        ('TabTransformer', TabTransformerConfig),
    ]

    # Dictionary to store evaluation results
    evaluation_results: Dict[str, Dict[str, Any]] = {}

    # Iterate over each model, train and evaluate
    for model_name, ModelConfigClass in model_configs:
        print(f"Training and evaluating {model_name}...")
        try:
            # Define configurations for PyTorch Tabular
            data_config = DataConfig(
                target=[target_col],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
                validation_split=0.2,  # Use 20% of training data for validation internally
            )

            # Model-specific configurations
            model_config = ModelConfigClass(
                task="regression",
                metrics=["mean_squared_error"],
                metrics_params=[{}],
            )

            trainer_config = TrainerConfig(
                max_epochs=100,
                batch_size=1024,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use accelerator for device selection
                devices=1 if torch.cuda.is_available() else None,  # Number of devices (GPUs/CPUs)
                early_stopping="valid_loss",
                early_stopping_patience=30,

            )

            # Instantiate the TabularModel
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=OptimizerConfig(),
                trainer_config=trainer_config,
            )

            # Train the model (no need to specify validation data, handled by `validation_split`)
            tabular_model.fit(train=full_df[full_df['split'] == 'train'])

            # Evaluate the model on each dataset split
            
            evaluation_dict = {}

            for split_name in ['train', 'test', 'lb']:
                split_df = full_df[full_df['split'] == split_name]
                y_true = split_df[target_col].values
                comb_id = split_df['comb_id']
                X_eval = split_df.drop(columns=[target_col, 'split', 'comb_id'])

                # Make predictions
                predictions = tabular_model.predict(X_eval)
                
                y_pred = predictions['target_prediction'].values.flatten()

                # Calculate weighted Pearson correlation
                weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)
                evaluation_dict[split_name] = {
                    'wpc': weighted_pear,
                    'pear_weights': pear_weights_df
                }

            



            evaluation_results[model_name] = evaluation_dict
            print(f"{model_name} - Train Weighted Pearson Correlation: {evaluation_dict['train']['wpc']:.4f}")
            print(f"{model_name} - Test Weighted Pearson Correlation: {evaluation_dict['test']['wpc']:.4f}")
            print(f"{model_name} - LB Weighted Pearson Correlation: {evaluation_dict['lb']['wpc']:.4f}\n")

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
