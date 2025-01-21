import time
import numpy as np
import random
import pandas as pd
from collections import Counter
from typing import Union, List, Tuple, Dict, Any
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, )
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from sklearn.exceptions import NotFittedError

import pytorch_lightning as pl


from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ModelConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer

# Suppress common warnings
import logging
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def freeze_categorical_embeddings(model):
    """
    Freezes the categorical embedding layers of the given model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param in module.parameters():
                param.requires_grad = False

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


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, categorical_features):
        self.model = model
        self.embedding_transformer = CategoricalEmbeddingTransformer(model)
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        # No fitting required
        return self

    def transform(self, X):
        # Check if the model and transformer are correctly initialized
        if not hasattr(self.embedding_transformer, '_mapping'):
            raise NotFittedError("The CategoricalEmbeddingTransformer is not fitted yet.")
        
        # Ensure that X contains the required categorical features
        missing_cols = set(self.categorical_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing categorical columns in input data: {missing_cols}")

        # Transform the categorical features into embeddings
        X_transformed = self.embedding_transformer.transform(X[self.categorical_features])
        # Extract the embedding columns
        embedding_cols = [col for col in X_transformed.columns if '_embed_dim_' in col]
        # Return the embeddings as a numpy array
        return X_transformed[embedding_cols].values


def train_evaluate_sklearn_pipeline(
    datasets: Dict[str, Dict[str, Any]],
    model: BaseEstimator,
    categorical_encoder: str = 'OneHotEncoder',
    model_path: str = None,
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
    categorical_encoder : str, optional
        'OneHotEncoder', 'LabelEncoder', or 'EmbeddingEncoder'. Determines which encoder to use for categorical features.
        Default is 'OneHotEncoder'.
    model_path : str, optional
        Path to the pretrained model checkpoint, required if categorical_encoder is 'EmbeddingEncoder'.
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

    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    continuous_features = X_train.select_dtypes(include=['number']).columns.tolist()

    if verbose:
        print("Categorical features:", categorical_features)
        print("Continuous features:", continuous_features)

    # Preprocessing pipelines for numeric and categorical features
    numeric_pipeline = Pipeline(steps=[
        ('selector', FunctionTransformer(lambda X: X[continuous_features], validate=False)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Choose the categorical transformer based on 'categorical_encoder'
    if categorical_encoder == 'LabelEncoder':
        categorical_pipeline = Pipeline(steps=[
            ('selector', FunctionTransformer(lambda X: X[categorical_features], validate=False)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = FeatureUnion(transformer_list=[
            ('num', numeric_pipeline),
            ('cat', categorical_pipeline)
        ])

    elif categorical_encoder == 'OneHotEncoder':
        categorical_pipeline = Pipeline(steps=[
            ('selector', FunctionTransformer(lambda X: X[categorical_features], validate=False)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = FeatureUnion(transformer_list=[
            ('num', numeric_pipeline),
            ('cat', categorical_pipeline)
        ])

    elif categorical_encoder == 'EmbeddingEncoder':
        # Ensure the categorical features in the data match those in the model
        missing_cols = set(categorical_features) - set(X_train.columns)
        if missing_cols:
            raise ValueError(f"Missing categorical columns in input data: {missing_cols}")

        # Create the EmbeddingTransformer
        tabular_model = TabularModel.load_model(model_path, map_location=torch.device("cpu"))
        embedding_transformer = EmbeddingTransformer(tabular_model, categorical_features)

        embedding_pipeline = Pipeline(steps=[
            ('selector', FunctionTransformer(lambda X: X[categorical_features], validate=False)),
            ('embedding_transformer', embedding_transformer)
        ])

        preprocessor = FeatureUnion(transformer_list=[
            ('num', numeric_pipeline),
            ('embedding', embedding_pipeline)
        ])

    else:
        raise ValueError(f"Invalid categorical_encoder: {categorical_encoder}. Choose 'OneHotEncoder', 'LabelEncoder', or 'EmbeddingEncoder'.")

    # Create the full pipeline
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

        # Ensure the dataset contains the required features
        if categorical_encoder == 'EmbeddingEncoder':
            missing_cols = set(categorical_features + continuous_features) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in {split_name} data: {missing_cols}")

        # Make predictions
        y_pred = model_pipeline.predict(X)

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)

        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }

    return evaluation_dict


def train_evaluate_pytorch_tabular_pipeline(
    datasets: Dict[str, pd.DataFrame],
    data_config: DataConfig,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    optimizer_config: OptimizerConfig = None,
    verbose: bool = True, 
    seed: int = 42, 
) -> Dict[str, Dict[str, Any]]:
    """
    Trains a PyTorch Tabular model using the provided configurations and datasets,
    and evaluates it using the weighted Pearson correlation coefficient.

    Parameters
    ----------
    datasets : dict
        A dictionary containing 'train', 'test', and 'lb' datasets, each with keys:
            - 'X': pandas.DataFrame of features.
            - 'y': pandas.Series of target values.
            - 'comb_id': pandas.Series of combination IDs.
    data_config : DataConfig
        Configuration for data handling in PyTorch Tabular.
    model_config : Any
        Configuration for the model in PyTorch Tabular.
    trainer_config : TrainerConfig
        Configuration for the trainer in PyTorch Tabular.
    optimizer_config : OptimizerConfig, optional
        Configuration for the optimizer in PyTorch Tabular. If None, default optimizer config is used.
    verbose : bool, optional
        If True, prints out progress information. Default is True.

    Returns
    -------
    evaluation_dict : dict
        A dictionary containing evaluation results for each dataset split ('train', 'test', 'lb'),
        with keys:
            - 'wpc': float, weighted Pearson correlation coefficient.
            - 'pear_weights': pandas.DataFrame, individual Pearson coefficients for each combination.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True) 

    if optimizer_config is None:
        optimizer_config = OptimizerConfig()

    train_df = datasets['train']['X']
    train_df['synergy_score'] = datasets['train']['y']

    test_df = datasets['test']['X']
    test_df['synergy_score'] = datasets['test']['y']

    lb_df = datasets['lb']['X']
    lb_df['synergy_score'] = datasets['lb']['y']

    
    if verbose:
        print("Data prepared for PyTorch Tabular.")

    # Instantiate the TabularModel
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        seed = seed
    )
    
    start_time = time.time()  # Start timing

    # Train the model (no need to specify validation data, handled by `validation_split`)
    tabular_model.fit(train=train_df, validation=test_df)
    training_time = time.time() - start_time

    # Dictionary to store evaluation results
    evaluation_dict = {}

    # Evaluate the model on each dataset split (train, test, leaderboard)
    for split_name, data in datasets.items():
        X_eval = data['X']
        y_true = data['y'].values
        comb_id = data['comb_id']

        # Make predictions
        predictions = tabular_model.predict(X_eval)

        # Extract predictions from the correct column
        if 'synergy_score_prediction' in predictions.columns:
            y_pred = predictions['synergy_score_prediction'].values.flatten()
        else:
            raise ValueError("Unexpected prediction structure.")

        # Calculate weighted Pearson correlation
        weighted_pear, pear_weights_df = weighted_pearson(comb_id, y_pred, y_true)
        evaluation_dict[split_name] = {
            'wpc': weighted_pear,
            'pear_weights': pear_weights_df
        }

    return evaluation_dict, tabular_model, training_time
