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


def weighted_pearson(
    comb_id_list: Union[List, np.ndarray, pd.Series],
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Tuple[float, pd.DataFrame]:
    """
    Computes the weighted Pearson correlation coefficient and a DataFrame of
    the individual Pearson coefficients for each combination.

    Parameters
    ----------
    comb_id_list : list, numpy.ndarray, or pandas.Series
        List of combination IDs.
    y_pred : numpy.ndarray
        Predicted synergy values.
    y_true : numpy.ndarray
        Ground truth synergy values.

    Returns
    -------
    weighted_pearson : float
        The weighted Pearson correlation coefficient.
    pearson_weights_df : pandas.DataFrame
        DataFrame containing the combination IDs, counts, and individual Pearson coefficients.
    """
    comb_id_arr = np.array(comb_id_list)

    if not (len(comb_id_arr) == len(y_pred) == len(y_true)):
        raise ValueError("comb_id_list, y_pred, and y_true must have the same length.")

    comb_id_counts = Counter(comb_id_arr)
    unique_comb_ids = list(comb_id_counts.keys())

    individual_pearsons = []
    numerator = 0.0
    denominator = 0.0

    for comb_id in unique_comb_ids:
        mask = (comb_id_arr == comb_id)
        n_samples = np.sum(mask)

        if n_samples < 2:
            rho_i = 0.0
        else:
            rho_i, _ = pearsonr(y_pred[mask], y_true[mask])
            if np.isnan(rho_i):
                rho_i = 0.0

        weight = np.sqrt(n_samples - 1)
        numerator += weight * rho_i
        denominator += weight

        individual_pearsons.append({
            "Combination ID": comb_id,
            "n_samples": n_samples,
            "Pearson coefficient": rho_i
        })

    if denominator == 0:
        weighted_pearson = np.nan
    else:
        weighted_pearson = numerator / denominator

    pearson_weights_df = pd.DataFrame(individual_pearsons)

    return weighted_pearson, pearson_weights_df

def train_evaluate_pipeline(
    datasets: Dict[str, Dict[str, Any]],
    model: BaseEstimator,
    categorical_encoder: str = 'OneHotEncoder',
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Trains a machine learning model using a preprocessing pipeline on the training data
    and evaluates it on the provided datasets using the weighted Pearson correlation coefficient.

    Parameters
    ----------
    datasets : dict
        A dictionary containing 'train', 'test', and 'lb' datasets, each with keys:
            - 'X': pandas.DataFrame of features.
            - 'y': pandas.Series of target values.
            - 'comb_id': pandas.Series of combination IDs.
    model : sklearn.base.BaseEstimator
        The machine learning model to be trained.
    categorical_encoder: str, optional
        'OneHotEncoder' or 'LabelEncoder'. Determines which encoder to use for categorical features.
        Default is 'OneHotEncoder'.
    verbose : bool, optional
        If True, prints out the list of categorical and continuous features. Default is True.

    Returns
    -------
    evaluation_dict : dict
        A dictionary containing evaluation results for each dataset split ('train', 'test', 'lb'),
        with keys:
            - 'wpc': float, weighted Pearson correlation coefficient.
            - 'pear_weights': pandas.DataFrame, individual Pearson coefficients for each combination.
    """
    # Extract training data
    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    train_comb_id = datasets['train']['comb_id']

    # Identify categorical and continuous features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    continuous_features = X_train.select_dtypes(include=['number']).columns.tolist()

    if verbose:
        print("Categorical features:", categorical_features)
        print("Continuous features:", continuous_features)

    # Preprocessing pipelines for numeric and categorical features
    

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Choose the categorical transformer based on 'categorical_encoder'
    if categorical_encoder == 'LabelEncoder':
        # Use OrdinalEncoder for features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    elif categorical_encoder == 'OneHotEncoder':
        # Use OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    else:
        raise ValueError(f"Invalid categorical_encoder: {categorical_encoder}. Choose 'OneHotEncoder' or 'LabelEncoder'.")



    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0.0  # Ensure output is dense
    )



    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model on the training data
    model_pipeline.fit(X_train, y_train)

    evaluation_dict = {}

    # Evaluate the model on each dataset split
    for split_name, data in datasets.items():
        X = data['X']
        y_true = data['y']
        comb_id = data['comb_id']

        # Make predictions
        y_pred = model_pipeline.predict(X)

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)

        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }

    return evaluation_dict