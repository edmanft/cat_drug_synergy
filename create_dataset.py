
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:49:05 2022

@author: Manuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
 
path = "C:/Users/Manuel/Desktop/universidad/master/TFM/datos/saezr_nc19_DataS1_modificado.xlsx"

sheet_to_df_map = pd.read_excel(path, sheet_name=None)

# %%

df_cell_lines = sheet_to_df_map["Cell lines"]

df_drug_comb = sheet_to_df_map["Drug combinations Ch1"]

df_drug_port = sheet_to_df_map["Drug portfolio"]

# %%


