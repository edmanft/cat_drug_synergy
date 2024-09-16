#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for TabNet Model.

This script loads the dataset, splits it into training, testing, and leaderboard sets,
and then trains and evaluates a TabNet model using a custom pipeline. The evaluation
results are collected and displayed in a sorted DataFrame.

Usage:
    python train_tabnet.py --data_path /path/to/data

    Ensure that the dataset paths are correctly specified before running the script.
"""

import os
import warnings
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

from src.data.process_data import load_dataset, split_sets
from src.model.evaluation import weighted_pearson

def main():
    """
    Main function to execute the training and evaluation of the TabNet model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for TabNet Model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    args = parser.parse_args()

    path_to_data = args.data_path

    # Suppress convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Define the file paths to your datasets
    drug_syn_path = os.path.join(path_to_data, 'drug_synergy.csv')
    cell_lines_path = os.path.join(path_to_data, 'cell_lines.csv')
    drug_portfolio_path = os.path.join(path_to_data, 'drug_portfolio.csv')

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
    full_dataset_df = load_dataset(drug_syn_path, cell_lines_path, drug_portfolio_path)

    # Split the dataset into training, testing, and leaderboard sets
    datasets = split_sets(full_dataset_df)

    # Extract training data
    X_train = datasets['train']['X'].values
    y_train = datasets['train']['y'].values
    train_comb_id = datasets['train']['comb_id']

    # Extract testing data
    X_test = datasets['test']['X'].values
    y_test = datasets['test']['y'].values
    test_comb_id = datasets['test']['comb_id']

    # Extract leaderboard data
    X_lb = datasets['lb']['X'].values
    y_lb = datasets['lb']['y'].values
    lb_comb_id = datasets['lb']['comb_id']

    # Initialize TabNet model
    model = TabNetRegressor()

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_name=['train', 'test'],
        eval_metric=['rmse'],
        max_epochs=1000,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # Evaluate the model
    evaluation_dict = {}

    for split_name, (X, y_true, comb_id) in zip(
        ['train', 'test', 'lb'],
        [(X_train, y_train, train_comb_id), (X_test, y_test, test_comb_id), (X_lb, y_lb, lb_comb_id)]
    ):
        # Make predictions
        y_pred = model.predict(X)

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)

        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }

        print(f"{split_name.capitalize()} Weighted Pearson Correlation: {weighted_pear:.4f}")

    # Collect results into lists for DataFrame creation
    split_list = []
    wpc_list = []

    for split_name, eval_dict in evaluation_dict.items():
        split_list.append(split_name)
        wpc_list.append(eval_dict['wpc'])

    # Create a DataFrame to display evaluation results
    evaluation_df = pd.DataFrame({
        'Split': split_list,
        'Weighted Pearson Correlation': wpc_list
    })

    # Sort the DataFrame based on the Weighted Pearson Correlation
    evaluation_df = evaluation_df.sort_values(by='Weighted Pearson Correlation', ascending=False)
    print("Evaluation Results:")
    print(evaluation_df.reset_index(drop=True))

if __name__ == '__main__':
    main()
