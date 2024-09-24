import numpy as np
import pandas as pd
from collections import Counter
from typing import Union, List, Tuple, Dict, Any
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer)
from sklearn.base import BaseEstimator

def load_dataset(
    drug_syn_path: str,
    cell_lines_path: str,
    drug_portfolio_path: str
) -> pd.DataFrame:
    # Load drug combinations data
    drug_synergy_df = pd.read_csv(drug_syn_path)
    # Filter out low-quality data
    drug_synergy_df = drug_synergy_df[drug_synergy_df['QA'] == 1]
    # Drop unnecessary columns
    drug_synergy_df.drop(columns=['Challenge', 'QA'], inplace=True)

    # Load cell lines data
    cell_lines_df = pd.read_csv(cell_lines_path)
    # Drop the COSMIC ID column
    cell_lines_df.drop(columns=['COSMIC ID'], inplace=True)

    # Load drug portfolio data
    drug_portfolio_df = pd.read_csv(drug_portfolio_path, sep='\t')
    # Clean up data by removing unwanted columns
    drug_portfolio_df.drop(columns=['Drug name'], inplace=True)

    # Merge datasets to form a comprehensive dataset
    full_dataset_df = pd.merge(drug_synergy_df, cell_lines_df, on='Cell line name', how='left')
    # Merge for Compound A
    full_dataset_df = pd.merge(full_dataset_df, drug_portfolio_df, left_on='Compound A', right_on='Challenge drug name', how='left', suffixes=('', '_A'))
    # Merge for Compound B
    full_dataset_df = pd.merge(full_dataset_df, drug_portfolio_df, left_on='Compound B', right_on='Challenge drug name', how='left', suffixes=('', '_B'))
    # Rename columns for clarity
    full_dataset_df.rename(columns={
        'Challenge drug name': 'Challenge drug name_A',
        'Putative target': 'Putative target_A',
        'Function': 'Function_A',
        'Pathway': 'Pathway_A',
        'HBA': 'HBA_A',
        'HBD': 'HBD_A',
        'Molecular weight': 'Molecular weight_A',
        'cLogP': 'cLogP_A',
        'Lipinski': 'Lipinski_A',
        'SMILES': 'SMILES_A'
    }, inplace=True)
    
    ## Redundant with Compound A and Compound B
    full_dataset_df.drop(columns=['Challenge drug name_A', 'Challenge drug name_B'], inplace=True)
    
     ## Redundant with GDSC tissue descriptor 2
    full_dataset_df.drop(columns=['GDSC tissue descriptor 1', 'TCGA label'], inplace=True)

    # More than 20% missing values
    full_dataset_df.drop(columns=['HBA_A', 'HBD_A', 'Molecular weight_A', 'cLogP_A', 'Lipinski_A',
       'SMILES_A', 'HBA_B', 'HBD_B', 'Molecular weight_B', 'cLogP_B',
       'Lipinski_B', 'SMILES_B'], inplace=True)


    return full_dataset_df

def split_dataset(
    full_dataset_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Splits the full dataset into training, testing, and leaderboard sets,
    sorting each by 'Combination ID', and separates features and target variables.

    Parameters
    ----------
    full_dataset_df : pandas.DataFrame
        The full dataset containing features, targets, and metadata.

    Returns
    -------
    datasets : dict
        A dictionary with keys 'train', 'test', and 'lb', each containing another
        dictionary with keys:
            - 'X': pandas.DataFrame of features.
            - 'y': pandas.Series of target values.
            - 'comb_id': pandas.Series of combination IDs.
    """
    # Split the dataset based on the 'Dataset' column
    train_data = full_dataset_df[full_dataset_df['Dataset'] == 'train'].copy()
    test_data = full_dataset_df[full_dataset_df['Dataset'] == 'test'].copy()
    lb_data = full_dataset_df[full_dataset_df['Dataset'] == 'LB'].copy()

    # Sort each dataset by 'Combination ID'
    train_data = train_data.sort_values(by='Combination ID', ascending=True)
    test_data = test_data.sort_values(by='Combination ID', ascending=True)
    lb_data = lb_data.sort_values(by='Combination ID', ascending=True)

    # Extract 'Combination ID' columns
    train_comb_id = train_data['Combination ID'].copy()
    test_comb_id = test_data['Combination ID'].copy()
    lb_comb_id = lb_data['Combination ID'].copy()

    # Drop 'Combination ID' columns from datasets
    train_data = train_data.drop(columns=['Combination ID'])
    test_data = test_data.drop(columns=['Combination ID'])
    lb_data = lb_data.drop(columns=['Combination ID'])

    # Separate features and target variables
    X_train = train_data.drop(columns=['Synergy score', 'Dataset'])
    y_train = train_data['Synergy score'].copy()

    X_test = test_data.drop(columns=['Synergy score', 'Dataset'])
    y_test = test_data['Synergy score'].copy()

    X_lb = lb_data.drop(columns=['Synergy score', 'Dataset'])
    y_lb = lb_data['Synergy score'].copy()

    # Return a dictionary instead of tuples
    datasets = {
        'train': {'X': X_train, 'y': y_train, 'comb_id': train_comb_id},
        'test': {'X': X_test, 'y': y_test, 'comb_id': test_comb_id},
        'lb': {'X': X_lb, 'y': y_lb, 'comb_id': lb_comb_id},
    }

    return datasets