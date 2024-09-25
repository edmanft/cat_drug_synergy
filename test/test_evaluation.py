import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.model.evaluation import weighted_pearson, train_evaluate_pipeline

def test_weighted_pearson():
    # Mock data
    comb_id_list = [1, 1, 2, 2]
    y_pred = np.array([0.5, 0.6, 0.7, 0.8])
    y_true = np.array([0.5, 0.6, 0.7, 0.8])

    # Calculate weighted Pearson
    weighted_pear, pearson_weights_df = weighted_pearson(comb_id_list, y_pred, y_true)

    # Assertions
    assert isinstance(weighted_pear, float)
    assert not pearson_weights_df.empty

def test_train_evaluate_pipeline():
    # Mock data
    data = {
        'Combination ID': [1, 2, 3],
        'Synergy score': [0.5, 0.6, 0.7],
        'Dataset': ['train', 'test', 'LB'],
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6]
    }
    full_dataset_df = pd.DataFrame(data)
    datasets = {
        'train': {'X': full_dataset_df[['Feature1', 'Feature2']], 'y': full_dataset_df['Synergy score'], 'comb_id': full_dataset_df['Combination ID']},
        'test': {'X': full_dataset_df[['Feature1', 'Feature2']], 'y': full_dataset_df['Synergy score'], 'comb_id': full_dataset_df['Combination ID']},
        'lb': {'X': full_dataset_df[['Feature1', 'Feature2']], 'y': full_dataset_df['Synergy score'], 'comb_id': full_dataset_df['Combination ID']}
    }

    # Model
    model = LinearRegression()

    # Train and evaluate
    evaluation_dict = train_evaluate_pipeline(datasets, model)

    # Assertions
    assert 'train' in evaluation_dict
    assert 'test' in evaluation_dict
    assert 'lb' in evaluation_dict
    assert 'wpc' in evaluation_dict['train']
    assert 'pear_weights' in evaluation_dict['train']
