#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:41:54 2021

@author: manuel
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Comment this if the data visualisations doesn't work on your side

plt.style.use('bmh')

# %%
path = "/home/manuel/Escritorio/universidad/master/tfm"
path += "/astra_zeneca/datos/ch1_train_combination_and_monoTherapy.csv"
df = pd.read_csv(path)
# %%
df.info()
print(df.shape)
"We can see that there are no null values, this dataset has been prepared"
cols = df.columns
# %%
"Let's take a look at the distribution of synergy scores"

print(df["SYNERGY_SCORE"].describe())

plt.figure(figsize=(13, 10))
sns.distplot(df['SYNERGY_SCORE'], color='g', bins=100, hist_kws={'alpha': 0.4})

sns.histplot(df['SYNERGY_SCORE'])

"We can see that the distribution is centered around zero and there is"
"no visible skedness. We don't need to make any transformation whatsoever"

# %%
"We separate all numerical and categorical values"

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.drop(columns = ["QA"], axis = 1, inplace = True)
print(df_num.head())

df_cat = df.select_dtypes(include = ["object"])
df_cat["QA"] = df["QA"]
print(df_cat.head())
# %%

"We checked that the dataset does not account for conmutativity, ie if"
"drug combination A-B is included, B-A is not included. We will engineer"
"it so that the resulting network is conmutative."


# %%

"Let's make histograms for all numerical variables"

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 
# %%

df_num_corr = df_num.corr()['SYNERGY_SCORE'][:-1 ] 
features_list = df_num_corr.sort_values(ascending=False)
print("These are the correlations with SYNERGY_SCORE:\n{}".format(features_list))

"""It appears as if all the numerical values make up for poor predictors
for SYNERGY_SCORE. Maybe we should check for an analysis """

# %%

for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['SYNERGY_SCORE'])

"We don't see anything here"

# %%


corr = df_num.drop('SYNERGY_SCORE', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.2) | (corr <= -0.2)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

# %%

plt.figure(figsize = (30, 26))
ax = sns.boxplot(x=df_cat['CELL_LINE'], y=df['SYNERGY_SCORE'])
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)

# %%

plt.figure(figsize = (15, 10))
ax = sns.boxplot(x=df_cat['COMPOUND_A'], y=df['SYNERGY_SCORE'])
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# %%



plt.figure(figsize = (15, 10))
ax = sns.boxplot(x=df_cat['COMPOUND_B'], y=df['SYNERGY_SCORE'])
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)

# %%

plt.figure(figsize = (10, 8))
ax = sns.boxplot(x=df_cat['QA'], y=df['SYNERGY_SCORE'])
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)

# %%

"""There are two outiers that are messing with the plots, let's
drop them at least for this analysis"""

print("With outliers:\n", df["SYNERGY_SCORE"].describe())

df_no_out = df[np.abs(df["SYNERGY_SCORE"])<500]
"If we take the cut at abs(SYNERGY_SCORE)<500 we just delete 3 points"
print("Without outliers:\n", df_no_out["SYNERGY_SCORE"].describe())

# %%
"Now we repeat the same plots"

def categorical_plots(dataframe, category_array, target_string):
    for col in category_array: 
        plt.figure(figsize = (20, 16))
        ax = sns.boxplot(x=dataframe[col], y=dataframe[target_string])
        plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
        plt.xticks(rotation=45)
category_array = ['CELL_LINE', "COMPOUND_A", "COMPOUND_B", "QA"]
categorical_plots(df_no_out, category_array, 
                  target_string = "SYNERGY_SCORE")
# %%
"Boxplot of the synergy score without outliers"
plt.figure(figsize=(13, 10))

sns.boxplot(df_no_out["SYNERGY_SCORE"])

""""Lastly we will check for the quality of the dataset"
which is given by the parameter QA"""
plt.figure(figsize=(10, 6))

sns.histplot(df["QA"].astype('category'))
plt.xlabel("QA", fontsize = 15)
plt.ylabel("Count", fontsize = 15)

plt.figure(figsize=(10, 6))

