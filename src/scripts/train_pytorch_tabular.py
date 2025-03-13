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
    python src/scripts/train_pytorch_tabular.py --data_path data --batch_size 256 --max_epoch 10 --es_patience 5 --no_embedding --seed 42
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating PyTorch Tabular Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument('--es_patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--no_embedding', action='store_true', help='Use OneHotEncoding instead of embeddings for categorical variables.')
    parser.add_argument('--morgan_fp', action='store_true', help='Whether to use Morgan fingerprints as features.')
    parser.add_argument('--fpSize', type=int, default=2048, help='The size of the Morgan fingerprints.')
    parser.add_argument('--fill_strategy', type=str, default='nan', help='Strategy to fill SMILES.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory where the models will be saved. If not provided, models will not be saved.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the evaluation results as a CSV file.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output.')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(args.seed, workers=True)


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
    
    if args.morgan_fp:
        smiles_path = os.path.join(args.data_path, 'drug_portfolio_smiles.csv')
        if not os.path.exists(smiles_path):
            print(f"Drug SMILES file not found at {smiles_path}")
            exit(1)
        
        # Load and process the dataset
        full_dataset_df, column_type_dict = load_dataset(drug_syn_path, cell_lines_path, 
                                          drug_portfolio_path, smiles_path=smiles_path, 
                                          fpSize=args.fpSize, fill_strategy=args.fill_strategy)
    else:
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
        #('TabNet', TabNetModelConfig),
        ('CategoryEmbedding', CategoryEmbeddingModelConfig),
        #('Node', NodeConfig),
        #('AutoInt', AutoIntConfig),
        #('FTTransformer', FTTransformerConfig),
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
                target=['synergy_score'],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
               )

            # Model-specific configurations
            model_config = ModelConfigClass(
                seed=args.seed,
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
                seed=args.seed,
                deterministic=True,
            
            )

        
            eval_dict, trained_model, training_time = train_evaluate_pytorch_tabular_pipeline(
                datasets=datasets,
                data_config=data_config,
                model_config=model_config,
                trainer_config=trainer_config,
                verbose=args.verbose, 
                seed = args.seed
            )


            

            evaluation_results[model_name] = eval_dict
            print(f"{model_name} - Training Time: {training_time:.2f} seconds")
            print(f"{model_name} - Train Weighted Pearson Correlation: {eval_dict['train']['wpc']:.4f}")
            print(f"{model_name} - Test Weighted Pearson Correlation: {eval_dict['test']['wpc']:.4f}")
            print(f"{model_name} - LB Weighted Pearson Correlation: {eval_dict['lb']['wpc']:.4f}\n")

            if args.model_dir is not None:
                model_save_path = os.path.join(args.model_dir, f"{model_name}_model.ckpt")
                trained_model.save_model(model_save_path)
                if args.verbose:
                    print(f"Model saved to: {model_save_path}")


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
        'LB WPC': lb_wpc_list,  
    })

    # Sort the DataFrame based on the Leaderboard Weighted Pearson Correlation
    evaluation_df = evaluation_df.sort_values(by='LB WPC', ascending=False)
    print("Evaluation Results:")
    print(evaluation_df.reset_index(drop=True))
    if args.save_path:
        evaluation_df.to_csv(args.save_path, index=False)
        print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()