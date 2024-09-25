#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization Script for TabNet Model using Optuna.

This script loads the dataset, splits it into training, testing, and leaderboard sets,
and then performs hyperparameter optimization for a TabNet model using Optuna, optimizing
the weighted Pearson correlation coefficient (WPC) on the test set.

Usage:
    python hyperparam_opt_tabnet.py --data_path path/to/data/ --verbose

Ensure that the dataset paths are correctly specified before running the script.
"""

import os
import warnings
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna
import torch

from src.data.process_data import load_dataset, split_dataset
from src.model.evaluation import weighted_pearson

def main():
    """
    Main function to execute the hyperparameter optimization for TabNet model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization Script for TabNet Model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for Optuna optimization.')
    parser.add_argument('--timeout', type=int, default=None, help='Time limit in seconds for the optimization.')
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

    # Prepare data
    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    train_comb_id = datasets['train']['comb_id']

    X_test = datasets['test']['X']
    y_test = datasets['test']['y']
    test_comb_id = datasets['test']['comb_id']

    # Identify categorical and continuous features
    categorical_features = column_type_dict['categorical']['col_names']
    categorical_idxs = column_type_dict['categorical']['col_idx']
    continuous_features = column_type_dict['numerical']['col_names']

    if args.verbose:
        print("Categorical features:", categorical_features)
        print("Continuous features:", continuous_features)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, continuous_features)
        ])

    # Fit preprocessor on training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Handle unknown categories
    categorical_dims = {}
    for col, idx in zip(categorical_features, categorical_idxs):
        # extract the dimension from Ordinal Encoder
        categorical_dims[col] = int(X_train[:, idx].max() + 2)  # +1 for zero-based indexing, +1 for unknown values
        # Adjust indices to be >=0
        X_train[:, idx] = X_train[:, idx] + 1
        X_test[:, idx] = X_test[:, idx] + 1

    cat_dims = [categorical_dims[col] for col in categorical_features]
    cat_emb_dim = [min(50, (cat_dim + 1) // 2) for cat_dim in cat_dims]

    # Prepare data for Optuna
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Objective function for Optuna
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_d': trial.suggest_int('n_d', 8, 64, step=8),
            'n_a': trial.suggest_int('n_a', 8, 64, step=8),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'mask_type': trial.suggest_categorical('mask_type', ['entmax', 'sparsemax']),
            'n_independent': trial.suggest_int('n_independent', 1, 5),
            'n_shared': trial.suggest_int('n_shared', 1, 5),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
            'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [64, 128, 256]), 
            'scheduler_type': trial.suggest_categorical('scheduler_type', ['StepLR', 'ExponentialLR', 'ReduceLROnPlateau']),
            }
        
        
        # Conditionally suggest scheduler hyperparameters
        scheduler_type = params['scheduler_type']
        if scheduler_type == 'StepLR':
            params['scheduler_step_size'] = trial.suggest_int('scheduler_step_size', 10, 50)
            params['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.1, 0.9)
            scheduler_fn = torch.optim.lr_scheduler.StepLR
            scheduler_params = dict(step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])
        elif scheduler_type == 'ExponentialLR':
            params['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.1, 0.9)
            scheduler_fn = torch.optim.lr_scheduler.ExponentialLR
            scheduler_params = dict(gamma=params['scheduler_gamma'])
        elif scheduler_type == 'ReduceLROnPlateau':
            params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 10, 50)
            params['scheduler_factor'] = trial.suggest_float('scheduler_factor', 0.1, 0.9)
            scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = dict(patience=params['scheduler_patience'], factor=params['scheduler_factor'])
        else:
            scheduler_fn = None
            scheduler_params = None


        # Initialize model
        model = TabNetRegressor(
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            cat_idxs=categorical_idxs,
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            lambda_sparse=params['lambda_sparse'],
            n_independent=params['n_independent'],
            n_shared=params['n_shared'],
            mask_type=params['mask_type'], 
            optimizer_params = dict(lr=params['lr']),
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
)

        # Train model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_test, y_test)],
            eval_name=['test'],
            eval_metric=['rmse'],
            max_epochs=300,
            patience=10,
            batch_size=params['batch_size'],
            virtual_batch_size=params['virtual_batch_size'],
        )

        # Evaluate on test set
        y_pred = model.predict(X_test).flatten()
        y_true = y_test.flatten()
        wpc_test, _ = weighted_pearson(test_comb_id, y_pred, y_true)
        # Return negative wpc because Optuna minimizes the objective
        return -wpc_test

    # Create study
    study = optuna.create_study(direction='minimize')
    # Optimize
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    # Get best hyperparameters
    best_params = study.best_params
    print(f"Best trial WPC: {-study.best_value}")
    print("Best hyperparameters:")
    print(best_params)
   
    # Retrieve the best scheduler and its parameters
   
    best_scheduler_type = best_params['scheduler_type']
    if best_scheduler_type == 'StepLR':
        scheduler_fn = torch.optim.lr_scheduler.StepLR
        scheduler_params = dict(step_size=best_params['scheduler_step_size'], gamma=best_params['scheduler_gamma'])
    elif best_scheduler_type == 'ExponentialLR':
        scheduler_fn = torch.optim.lr_scheduler.ExponentialLR
        scheduler_params = dict(gamma=best_params['scheduler_gamma'])
    elif best_scheduler_type == 'ReduceLROnPlateau':
        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = dict(patience=best_params['scheduler_patience'], factor=best_params['scheduler_factor'])
    else:
        scheduler_fn = None
        scheduler_params = None

    # Retrain model with best hyperparameters
    best_model = TabNetRegressor(
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            cat_idxs=categorical_idxs,
            n_d=best_params['n_d'],
            n_a=best_params['n_a'],
            n_steps=best_params['n_steps'],
            gamma=best_params['gamma'],
            lambda_sparse=best_params['lambda_sparse'],
            n_independent=best_params['n_independent'],
            n_shared=best_params['n_shared'],
            mask_type=best_params['mask_type'], 
            optimizer_params= dict(lr=best_params['lr']),
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
    )

    # Re-adjust the categorical indices for the full dataset
    for split_name in datasets.keys():
        data = datasets[split_name]
        X = data['X']
        X = preprocessor.transform(X)
        # Adjust indices to be >=0
        for idx in range(len(categorical_features)):
            X[:, idx] = X[:, idx] + 1
        datasets[split_name]['X_transformed'] = X
        datasets[split_name]['y'] = data['y'].values.reshape(-1, 1)

    # Retrain on full training data
    X_train_full = datasets['train']['X_transformed']
    y_train_full = datasets['train']['y']

    X_valid = datasets['test']['X_transformed']
    y_valid = datasets['test']['y']

    

    best_model.fit(
    X_train=X_train_full,
    y_train=y_train_full,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    eval_metric=['rmse'],
    max_epochs=1000,
    patience=50,
    batch_size=best_params['batch_size'],
    virtual_batch_size=best_params['virtual_batch_size'],
    )


    # Evaluate the model on each dataset split
    evaluation_dict = {}
    for split_name, data in datasets.items():
        X = data['X_transformed']
        y_true = data['y'].flatten()
        comb_id = data['comb_id']

        # Make predictions
        y_pred = best_model.predict(X).flatten()

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)

        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }

    print("Evaluation results:")
    for split_name in evaluation_dict:
        print(f"{split_name} WPC: {evaluation_dict[split_name]['wpc']}")

    # Output the evaluation_dict and hyperparameter dictionary
    print("Best hyperparameters:")
    print(best_params)
    print(f"WPC: {evaluation_dict['lb']['wpc']}")


if __name__ == '__main__':
    main()
