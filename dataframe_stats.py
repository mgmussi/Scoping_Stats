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

df = pd.read_excel('__Dataframe_ScopingReview.xlsx', sheet_name=1, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
# print(df)
# print(df['ID'].values)
# print(df['On_ACC'].values)
sns.set_style("darkgrid")
idx = (df['On_ACC'].values != '-')

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

##combine arrays and use np.unique(x) to get all values that dont have - and are Hom or Het
idx2 = (df['_Div_input'].values == 'Homogeneous' and df['On_ACC'].values != '-') 
idx3 = (df['_Div_input'].values == 'Heterogeneous' and df['On_ACC'].values != '-')
plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx2], y = df['On_ACC'][idx2].astype(float), 
               hue = df['_Div_input_sp'][idx2], split = False, data = df)
plt.show()

plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx3], y = df['On_ACC'][idx3].astype(float), 
               hue = df['_Div_input_sp'][idx3], split = False, data = df)
plt.show()