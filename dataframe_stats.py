#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:41:45 2022

@author: Matheus_Mussi
"""
import pandas as pd

df = pd.read_excel('__Dataframe_ScopingReview.xlsx', sheet_name=1, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
print(df)
print(df['ID'].values)
print(df['On_ACC'].values)

#loop through values and take '-' out