import os
import numpy as np
import pandas as pd

# Define the directory path and number of seeds
path = 'benchmarks'
nseeds = 20

# Define function to load and aggregate DataFrames over seeds
def load_and_aggregate(file_base_name, column_prefix):
    dfs = []
    for seed in range(nseeds):
        file_name = f"{file_base_name}_seed_{seed}.csv"
        df = pd.read_csv(os.path.join(path, file_name))
        df = df[['Model', 'LB WPC']]
        dfs.append(df)
    # Concatenate all DataFrames
    full_df = pd.concat(dfs)
    # Group by 'Model' and compute mean and std
    agg_df = full_df.groupby('Model')['LB WPC'].agg(['mean', 'std']).reset_index()
    # Format the mean and std
    agg_df['mean'] = agg_df['mean'].apply(lambda x: f"{x:.3f}")
    agg_df['std'] = agg_df['std'].apply(lambda x: f"{x:.3f}")
    # Rename columns with the specified prefix
    agg_df = agg_df.rename(columns={
        'mean': f"{column_prefix}_mean",
        'std': f"{column_prefix}_std"
    })
    return agg_df

# Load and process PyTorch results
pt_results = load_and_aggregate('pytorch_tabular', 'PyTorchTabular')
pt_results.to_csv(os.path.join(path, 'pytorch_tabular_pp.csv'), index=False)

# Load and prepare each encoding type DataFrame
label_encoder_df = load_and_aggregate('sklearn_LabelEncoder', 'LabelEncoder')
onehot_encoder_df = load_and_aggregate('sklearn_OneHotEncoder', 'OneHotEncoder')
autoint_df = load_and_aggregate('sklearn_EmbeddingAutoInt', 'AutoInt')
category_embedding_df = load_and_aggregate('sklearn_EmbeddingCategoryEmbedding', 'CategoryEmbedding')
tabtransformer_df = load_and_aggregate('sklearn_EmbeddingTabTransformer', 'TabTransformer')

# Merge all encoding DataFrames into a single DataFrame
sklearn_encodings_df = (
    label_encoder_df
    .merge(onehot_encoder_df, on='Model', how='inner')
    .merge(autoint_df, on='Model', how='inner')
    .merge(category_embedding_df, on='Model', how='inner')
    .merge(tabtransformer_df, on='Model', how='inner')
)

sklearn_encodings_df.to_csv(os.path.join(path, 'sklearn_encodings_pp.csv'), index=False)
