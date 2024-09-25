import pytest
import pandas as pd
from src.data.process_data import load_dataset, split_dataset

def test_load_dataset():
    # Mock data
    drug_syn_data = {
        'QA': [1, 1],
        'Challenge': ['A', 'B'],
        'Cell line name': ['Cell1', 'Cell2'],
        'Compound A': ['DrugA1', 'DrugA2'],
        'Compound B': ['DrugB1', 'DrugB2']
    }
    drug_syn_df['GDSC tissue descriptor 1'] = ['desc1', 'desc2']
    drug_syn_df['TCGA label'] = ['label1', 'label2']
    cell_lines_data = {
        'Cell line name': ['Cell1', 'Cell2'],
        'COSMIC ID': [123, 456]
    }
    drug_portfolio_data = {
        'Challenge drug name': ['DrugA1', 'DrugB1'],
        'Drug name': ['NameA1', 'NameB1']
    }

    # Convert to DataFrames
    drug_syn_df = pd.DataFrame(drug_syn_data)
    cell_lines_df = pd.DataFrame(cell_lines_data)
    drug_portfolio_df = pd.DataFrame(drug_portfolio_data)

    # Save to CSV
    drug_syn_df.to_csv('drug_syn.csv', index=False)
    cell_lines_df.to_csv('cell_lines.csv', index=False)
    drug_portfolio_df.to_csv('drug_portfolio.csv', sep='\t', index=False)

    # Load dataset
    full_dataset_df, column_type_dict = load_dataset('drug_syn.csv', 'cell_lines.csv', 'drug_portfolio.csv')

    # Assertions
    assert not full_dataset_df.empty
    assert 'Cell line name' in full_dataset_df.columns
    assert 'Compound A' in full_dataset_df.columns
    assert 'Compound B' in full_dataset_df.columns

def test_split_dataset():
    # Mock data
    data = {
        'Combination ID': [1, 2, 3],
        'Synergy score': [0.5, 0.6, 0.7],
        'Dataset': ['train', 'test', 'LB'],
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6]
    }
    full_dataset_df = pd.DataFrame(data)

    # Split dataset
    datasets = split_dataset(full_dataset_df)

    # Assertions
    assert 'train' in datasets
    assert 'test' in datasets
    assert 'lb' in datasets
    assert not datasets['train']['X'].empty
    assert not datasets['test']['X'].empty
    assert not datasets['lb']['X'].empty
