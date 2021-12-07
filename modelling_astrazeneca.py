#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:13:47 2021

@author: manuel
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error 
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_tree
from matplotlib.pylab import rcParams
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import catboost
# %%

path = "/home/manuel/Escritorio/universidad/master/tfm"
path += "/astra_zeneca/datos/ch1_train_combination_and_monoTherapy.csv"
df = pd.read_csv(path)

X = df[['CELL_LINE', 'COMPOUND_A', 'COMPOUND_B', 'MAX_CONC_A', 'MAX_CONC_B',
       'IC50_A', 'H_A', 'Einf_A', 'IC50_B', 'H_B', 'Einf_B', 'QA']]
y = df["SYNERGY_SCORE"]
# %%

"""First we need to do one hot encoding, we will do this using pandas
dummy variable
"""
ONE_HOT_COLS = ["CELL_LINE", 'COMPOUND_A', 'COMPOUND_B','QA']
print("Starting DF shape: %d, %d" % df.shape)

X_encoded = X.copy()
for col in ONE_HOT_COLS:
    s = X_encoded[col].unique()

    # Create a One Hot Dataframe with 1 row for each unique value
    one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
    one_hot_df[col] = s

    print("Adding One Hot values for %s (the column has %d unique values)" % (col, len(s)))
    pre_len = len(X_encoded)

    # Merge the one hot columns
    X_encoded = X_encoded.merge(one_hot_df, on=[col], how="left")
    assert len(X) == pre_len
    print(X_encoded.shape)
X_encoded.drop(columns = ONE_HOT_COLS, inplace = True)
# %%

"Create train and validation splits"
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2,
                                              random_state=0)

def XGBoost_mse(X_train, X_val, y_train, y_val):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred, squared=True )
    return mse

mse = XGBoost_mse(X_train, X_val, y_train, y_val)
print(f"MSE  : {mse}")

# %%

m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(X_train, y_train)
y_pred = m.predict(X_val)
mse = mean_squared_error(y_val, y_pred, squared=True )
print(f"MSE  : {mse}")


# %%
"let's try catboost"

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                              random_state=0)
cat_features_index= [0,1,2,11]


clf = catboost.CatBoostRegressor()

clf.fit(X_train, y_train, cat_features = cat_features_index)
y_pred = clf.predict(X_val)
mse = mean_squared_error(y_val, y_pred, squared=True )
print(f"MSE  : {mse}")
