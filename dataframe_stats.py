#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:41:45 2022

@author: Matheus_Mussi
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('__Dataframe_ScopingReview.xlsx', sheet_name=1, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
# print(df)
# print(df['ID'].values)
# print(df['On_ACC'].values)

for index,row in df.iterrows():
    if row['On_ACC'] != '-':
        plt.plot(int(row['ID']), float(row['On_ACC']), 'k.-')
    

plt.show()