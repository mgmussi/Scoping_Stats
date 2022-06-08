#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:41:45 2022

@author: Matheus_Mussi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_excel('__Dataframe_ScopingReview.xlsx', sheet_name=1, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
# print(df)
# print(df['ID'].values)
# print(df['On_ACC'].values)
sns.set_style("darkgrid")
idx = (df['On_ACC'].values != '-')
idx_acc = [counter for counter, value in enumerate(np.array(df['On_ACC'])) if value != '-']

## Acc per paper
for index,row in df.iterrows():
    if row['On_ACC'] != '-':
        plt.plot(int(row['ID']), float(row['On_ACC']), 'k.-')
    
sns.rugplot(data = df, x = df['ID'][idx], y = df['On_ACC'][idx])
plt.show()


## Barplot for Papers/Years
plt.figure()
sns.countplot(df['Year'].astype(int), palette = "flare")
plt.ylim([0,18])
plt.show()


## Violin plot
idx_het = [counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Heterogeneous']
idx_hom = [counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Homogeneous']

# idx_hom_acc = np.unique(np.array(idx_hom + idx_acc))
idx_hom_acc = list(set.intersection(set(idx_hom), set(idx_acc)))
# idx_het_acc = np.unique(np.array(idx_het + idx_acc))
idx_het_acc = list(set.intersection(set(idx_het), set(idx_acc)))

plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_hom_acc], y = df['On_ACC'][idx_hom_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_hom_acc], split = True, data = df)
plt.show()

plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_het_acc], y = df['On_ACC'][idx_het_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_het_acc], split = True, data = df)
plt.show()