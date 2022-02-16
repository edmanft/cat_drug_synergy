#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:58:10 2022

@author: manuel
"""

import numpy as np
import pandas as pd

# %%


path = "/home/manuel/Escritorio/universidad/master/tfm/codigos/expand_dataset/cell_line_info.csv"

cell_df = pd.read_csv(path, delimiter = ",")

path2 = "/home/manuel/Escritorio/universidad/master/tfm/codigos/expand_dataset/drug_comb.csv"

dc_df = pd.read_csv(path2, delimiter = ",")
# %%

zip_iterator = zip(cell_df['Cell line name'], 
                   cell_df[['GDSC tissue descriptor 1',
                            'GDSC tissue descriptor 2', 'TCGA label']])

cell_dict = dict(zip_iterator)

# %%

cell_name = list(cell_df['Cell line name'])

cell_info = cell_df[['GDSC tissue descriptor 1',
         'GDSC tissue descriptor 2', 'TCGA label']]

cell_info = np.asarray(cell_info)


# %%

zip_iterator = zip(cell_name, cell_info)

cell_dict = dict(zip_iterator)

# %%
def aux(x):
    
    return pd.Series([cell_dict[x, 0], cell_dict[x, 1], cell_dict[x, 2]])

# %%




dc_df[['GDSC 1', 'GDSC 2', 'GDSC 3']] = dc_df['Cell line name'].apply(lambda x: pd.Series([cell_dict[x][0], cell_dict[x][1], cell_dict[x][2]]))











