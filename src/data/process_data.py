import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from collections import Counter
from typing import Union, List, Tuple, Dict, Any
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer)
from sklearn.base import BaseEstimator

def smiles_to_morgan_fp(smiles_df, radius=4,fpSize=2048):
    morgan_fp_dict = {}
    fail_count = 0
    for i in tqdm(range(len(smiles_df))):
        drug_name = smiles_df.iloc[i]['Challenge drug name']
        smiles = smiles_df.iloc[i]['SMILES']
        
        try: 
            mol = Chem.MolFromSmiles(smiles)
            morgan_fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
            morgan_fp = morgan_fp.GetFingerprint(mol)
            morgan_fp = np.array(morgan_fp)
            morgan_fp_dict[drug_name] = morgan_fp
        except: # save numpy arrays of nans
            morgan_fp_dict[drug_name] = np.full(fpSize, 0)
            fail_count += 1
    print(f"Failed to convert {fail_count}/{len(smiles_df)} SMILES strings to Morgan fingerprints")
    print(f"Setting them to all-zero arrays")
            
    return morgan_fp_dict

def load_dataset(
    drug_syn_path: str,
    cell_lines_path: str,
    drug_portfolio_path: str, 
    smiles_path: str|None = None
) -> pd.DataFrame:
    # Load drug combinations data
    drug_synergy_df = pd.read_csv(drug_syn_path)
    # Filter out low-quality data
    drug_synergy_df = drug_synergy_df[drug_synergy_df['QA'] == 1]
    # Drop unnecessary columns
    drug_synergy_df.drop(columns=['Challenge', 'QA'], inplace=True)
    # Check and remove duplicates
    if drug_synergy_df.duplicated().sum() > 0:
        print(f"Found {drug_synergy_df.duplicated().sum()} duplicate rows in drug_synergy_df. Removing them.")
        drug_synergy_df.drop_duplicates(inplace=True)
    

    # Load cell lines data
    cell_lines_df = pd.read_csv(cell_lines_path)
    # Select only the AZ-DREAM cell lines
    cell_lines_df = cell_lines_df[cell_lines_df['AZ-DREAM '] == 1]

    # Drop the COSMIC ID and datasets columns
    cell_lines_df.drop(columns=['COSMIC ID', 'AZ-DREAM ', "O\'Neil et al. 2016"], inplace=True)
    # Check and remove duplicates
    if cell_lines_df.duplicated().sum() > 0:
        print(f"Found {cell_lines_df.duplicated().sum()} duplicate rows in cell_lines_df. Removing them.")
        cell_lines_df.drop_duplicates(inplace=True)

    # Load drug portfolio data
    drug_portfolio_df = pd.read_csv(drug_portfolio_path, sep='\t')
    # Clean up data by removing unwanted columns
    drug_portfolio_df.drop(columns=['Drug name'], inplace=True)
    # Check and remove duplicates
    if drug_portfolio_df.duplicated().sum() > 0:
        print(f"Found {drug_portfolio_df.duplicated().sum()} duplicate rows in drug_portfolio_df. Removing them.")
        drug_portfolio_df.drop_duplicates(inplace=True)

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
    
    if smiles_path is not None:
        smiles_fp_df = pd.read_csv(smiles_path, delimiter='\t')
        fpSize=2048
        morgan_fp_dict = smiles_to_morgan_fp(smiles_fp_df, radius=4, fpSize=fpSize)
        full_dataset_df.copy()
        fp_columns_a = [f'MorganFP_A_{i}' for i in range(fpSize)]
        fp_columns_b = [f'MorganFP_B_{i}' for i in range(fpSize)]

        full_dataset_df[fp_columns_a] = full_dataset_df['Compound A'].map(morgan_fp_dict).apply(pd.Series)
        full_dataset_df[fp_columns_b] = full_dataset_df['Compound B'].map(morgan_fp_dict).apply(pd.Series)

    else:
        print("No SMILES fingerprints provided. Skipping the step.")
        

    # Define categorical and numerical columns
    categorical_columns = [
        'Cell line name', 'Compound A', 'Compound B', 'GDSC tissue descriptor 2', 
        'MSI', 'Growth properties', 'Putative target_A', 'Function_A', 'Pathway_A', 
        'Putative target_B', 'Function_B', 'Pathway_B'
    ]
    numerical_columns = [
        'Max. conc. A', 'IC50 A', 'H A', 'Einf A', 
        'Max. conc. B', 'IC50 B', 'H B', 'Einf B'
    ]
    if smiles_path is not None:
        numerical_columns += fp_columns_a + fp_columns_b

    # Define columns that will not be used in training but are needed for future processing steps
    not_training_columns = ['Synergy score', 'Combination ID', 'Dataset']

    # Reorder the columns: categorical first, then numerical
    full_dataset_df = full_dataset_df[categorical_columns + numerical_columns + not_training_columns]

    # Check and assert no duplicates in the final DataFrame
    if full_dataset_df.duplicated().sum() > 0:
        print(f"Found {full_dataset_df.duplicated().sum()} duplicate rows in the final full_dataset_df.")
    assert full_dataset_df.duplicated().sum() == 0, "There are duplicates in the final dataset!"

    # Create a dictionary with column types and their indices
    column_type_dict = {
        'categorical': {
            'col_names': categorical_columns,
            'col_idx': [full_dataset_df.columns.get_loc(col) for col in categorical_columns]
        },
        'numerical': {
            'col_names': numerical_columns,
            'col_idx': [full_dataset_df.columns.get_loc(col) for col in numerical_columns]
        }
    }

    return full_dataset_df, column_type_dict

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