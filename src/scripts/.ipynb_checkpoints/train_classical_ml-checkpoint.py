#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Evaluating Multiple Regression Models.

This script loads the dataset, splits it into training, testing, and leaderboard sets,
and then iterates over a list of regression models to train and evaluate them using
a custom pipeline. The evaluation results are collected and displayed in a sorted DataFrame.

Usage:
    python script_name.py --data_path path/to/data/ --encoder OneHotEncoder

    Ensure that the dataset paths are correctly specified before running the script.
"""

import os
import sys
sys.path.append('.')
import warnings
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

from src.data.process_data import load_dataset, split_dataset
from src.model.evaluation import train_evaluate_sklearn_pipeline

def main():
    """
    Main function to execute the training and evaluation of multiple regression models.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating Multiple Regression Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--encoder', type=str, choices=['LabelEncoder', 'OneHotEncoder', 'EmbeddingEncoder'], 
                    default='OneHotEncoder', help='The type of categorical encoder to use.')
    parser.add_argument('--morgan_fp', action='store_true', help='Whether to use Morgan fingerprints as features.')
    parser.add_argument('--fpSize', type=int, default=2048, help='The size of the Morgan fingerprints.')
    parser.add_argument('--fill_strategy', type=str, default='nan', help='Strategy to fill SMILES.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pretrained model checkpoint (only required for EmbeddingEncoder).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the evaluation results as a CSV file.')
    args = parser.parse_args()

    # Ensure model_path is provided if encoder is EmbeddingEncoder
    if args.encoder == 'EmbeddingEncoder' and not args.model_path:
        parser.error("--model_path is required when --encoder is set to 'EmbeddingEncoder'.")


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
        full_dataset_df, _ = load_dataset(drug_syn_path, cell_lines_path, 
                                          drug_portfolio_path, smiles_path=smiles_path, 
                                          fpSize=args.fpSize, fill_strategy=args.fill_strategy)
    else:
        # Load and process the dataset
        full_dataset_df, _ = load_dataset(drug_syn_path, cell_lines_path, drug_portfolio_path)

    # Split the dataset into training, testing, and leaderboard sets
    datasets = split_dataset(full_dataset_df)

    # Define the list of regression models to evaluate
    np.random.seed(args.seed)
    models = [
    # Linear Models
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet Regression', ElasticNet()),
    ('Bayesian Ridge Regression', BayesianRidge()),
    ('Stochastic Gradient Descent', SGDRegressor(max_iter=1000, tol=1e-3, random_state=args.seed)),
    
    # Tree-based Models
    ('Decision Tree', DecisionTreeRegressor(random_state=args.seed)),
    ('Random Forest', RandomForestRegressor(random_state=args.seed)),
    ('Extra Trees', ExtraTreesRegressor(random_state=args.seed)),
    
    # Boosting Models
    ('AdaBoost', AdaBoostRegressor(random_state=args.seed)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=args.seed)),
    ('XGBRegressor', XGBRegressor(random_state=args.seed)),
    
    # Kernel-based and Non-linear Models
    ('Support Vector Regression', SVR()),
    ('Gaussian Process', GaussianProcessRegressor())
    ]

    # Dictionary to store evaluation results
    evaluation_results: Dict[str, Dict[str, Any]] = {}

    # Iterate over each model, train and evaluate
    for name, model in models:
        print(f"Training and evaluating {name}...")
        try:
            eval_dict = train_evaluate_sklearn_pipeline(
                datasets=datasets,
                model=model,
                categorical_encoder=args.encoder, 
                model_path=args.model_path,
                verbose=False
            )
            evaluation_results[name] = eval_dict
            print(f"{name} - Train Weighted Pearson Correlation: {eval_dict['train']['wpc']:.4f}")
            print(f"{name} - Test Weighted Pearson Correlation: {eval_dict['test']['wpc']:.4f}")
            print(f"{name} - LB Weighted Pearson Correlation: {eval_dict['lb']['wpc']:.4f}\n")
        except Exception as e:
            print(f"Error training {name}: {e}\n")

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

   
    print("Evaluation Results:")
    print(evaluation_df.reset_index(drop=True))
    if args.save_path:
        evaluation_df.to_csv(args.save_path, index=False)
        print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()
