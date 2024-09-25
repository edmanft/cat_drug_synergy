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
from pytorch_tabnet.tab_model import TabNetRegressor

from src.data.process_data import load_dataset, split_dataset
from src.model.evaluation import train_evaluate_tabnet

def main():
    """
    Main function to execute the training and evaluation of multiple regression models.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training Script for Evaluating Multiple Regression Models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing data files.')
    parser.add_argument('--encoder', type=str, choices=['LabelEncoder', 'OneHotEncoder'], default='OneHotEncoder', 
        help='The type of categorical encoder to use.')
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

    # Define the list of regression models to evaluate
    models = [
        ('TabNetRegressor', TabNetRegressor()),
        ]
    print(datasets)
    print(datasets['train']['X'].columns)
    print(column_type_dict)


if __name__ == '__main__':
    main()
