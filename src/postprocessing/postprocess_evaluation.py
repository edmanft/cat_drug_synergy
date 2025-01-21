import os
import numpy as np
import pandas as pd

# Define the directory path
path = 'figures/ms/tables'

# Load and process PyTorch results
pt_results = pd.read_csv(os.path.join(path, 'pytorch_tabular.csv'))[['Model', 'LB WPC']]
pt_results['LB WPC'] = pt_results['LB WPC'].apply(lambda x: f"{x:.3f}")
pt_results.to_csv(os.path.join(path, 'pytorch_tabular_pp.csv'), index=False)

# Define function to load and prepare individual encoding DataFrames
def load_and_select(file_name, column_name):
    df = pd.read_csv(os.path.join(path, file_name))
    df = df[['Model', 'LB WPC']].rename(columns={'LB WPC': column_name})
    df[column_name] = df[column_name].apply(lambda x: f"{x:.3f}")
    return df

# Load and prepare each encoding type DataFrame
label_encoder_df = load_and_select('sklearn_LabelEncoder.csv', 'LabelEncoder')
onehot_encoder_df = load_and_select('sklearn_OneHotEncoder.csv', 'OneHotEncoder')
autoint_df = load_and_select('sklearn_EmbeddingAutoInt.csv', 'AutoInt')
category_embedding_df = load_and_select('sklearn_EmbeddingCategoryEmbedding.csv', 'CategoryEmbedding')
tabtransformer_df = load_and_select('sklearn_EmbeddingTabTransformer.csv', 'TabTransformer')


# Merge all encoding DataFrames into a single DataFrame
sklearn_encodings_df = (
    label_encoder_df
    .merge(onehot_encoder_df, on='Model', how='inner')
    .merge(autoint_df, on='Model', how='inner')
    .merge(category_embedding_df, on='Model', how='inner')
    .merge(tabtransformer_df, on='Model', how='inner')
)

sklearn_encodings_df.to_csv(os.path.join(path, 'sklearn_encodings_pp.csv'), index=False)

