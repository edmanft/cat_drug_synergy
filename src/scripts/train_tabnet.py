#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for TabNet Model.

This script loads the dataset, splits it into training, testing, and leaderboard sets,
and then trains and evaluates a TabNet model using a custom pipeline. The evaluation
results are collected and displayed in a sorted DataFrame.

Usage:
    python script_name.py --data_path path/to/data/ --encoder OneHotEncoder

    Ensure that the dataset paths are correctly specified before running the script.
"""

import os
import warnings
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer)
from pytorch_tabnet.tab_model import TabNetRegressor

from src.data.process_data import load_dataset, split_dataset
from src.model.evaluation import weighted_pearson

def main():
    """
    Main function to execute the training and evaluation of multiple regression models.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating Multiple Regression Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output.')

    #parser.add_argument('--encoder', type=str, choices=['LabelEncoder', 'OneHotEncoder'], default='OneHotEncoder', 
    #    help='The type of categorical encoder to use.')
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
    
    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    train_comb_id = datasets['train']['comb_id']

    # Identify categorical and continuous features
    categorical_features = column_type_dict['categorical']['col_names']
    categorical_idxs = column_type_dict['categorical']['col_idx']
    continuous_features = column_type_dict['numerical']['col_names']

    if args.verbose:
        print("Categorical features:", categorical_features)
        print("Continuous features:", continuous_features)

    

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


    X_train = preprocessor.fit_transform(X_train)
    
    X_test = datasets['test']['X']
    X_test = preprocessor.transform(X_test)
    y_test = datasets['test']['y']
    
    categorical_dims = {}

    for col, idx in zip(categorical_features, categorical_idxs):
        # extract the dimension from Ordinal Encoder
        categorical_dims[col] = int(X_train[:, idx].max() + 2) # +1 for zero-based indexing, +1 for unknown values
        # so that unknown values are not -1 (Tabnet issues)
        X_train[:, idx] = X_train[:, idx] + 1 
        X_test[:, idx] = X_test[:, idx] + 1 
    cat_dims = [categorical_dims[col] for col in categorical_features]
    cat_emb_dim = [min(50, (cat_dim + 1) // 2) for cat_dim in cat_dims]
    
    model = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=categorical_idxs)
    
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    
    model.fit(X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['test'], eval_metric=['rmse'],
        max_epochs=1000, patience=50, batch_size=1024, virtual_batch_size=128)
    print(f"Tabnet model trained successfully.")

    
    # Evaluate the model on each dataset split
    evaluation_dict = {}
    for split_name, data in datasets.items():
        X = data['X']
        X = preprocessor.transform(X)
        # check if there is unknown categories
        for idx, col in enumerate(categorical_features):
            X[:, idx] = X[:, idx] + 1 
        y_true = data['y'].values
        comb_id = data['comb_id']


        # Make predictions
        y_pred = model.predict(X).flatten()

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)

        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }
    
    print(evaluation_dict)

if __name__ == '__main__':
    main()
