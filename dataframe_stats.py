#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:41:45 2022

@author: Matheus_Mussi
"""
import ColorMap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import random

ColorMap
sns.set_style("darkgrid")

cwd = os.getcwd()+'/__Dataframe_ScopingReview.xlsx'
df = pd.read_excel(cwd, sheet_name=1, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
# print(df)
# print(df['ID'].values)
# print(df['On_ACC'].values)

idx = (df['On_ACC'].values != '-')
idx_acc = [counter for counter, value in enumerate(np.array(df['On_ACC'])) if value != '-']
idx_age = [counter for counter, value in enumerate(np.array(df['Pop_age'])) if value != '-']
idx_age_to_ = [counter for counter, value in enumerate(np.array(df['Pop_age'].astype(str))) if 'to' in value]
idx_age_ = [counter for counter, value in enumerate(np.array(df['Pop_age'].astype(str))) if not('to' in value)]
idx_age_sv = [counter for counter, value in enumerate(np.array(df['Pop_age'][idx_age].astype(str))) if not('-' in value)]
idx_DivIn = [[counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Heterogeneous'],
             [counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Homogeneous']]


idx_het_acc = list(set.intersection(set(idx_DivIn[0]), set(idx_acc)))
idx_hom_acc = list(set.intersection(set(idx_DivIn[1]), set(idx_acc)))
idx_age_single = list(set.intersection(set(idx_age), set(idx_age_), set(idx_acc)))
idx_age_range = list(set.intersection(set(idx_age), set(idx_age_to_), set(idx_acc)))
###############################################################################
################################## FUNCTIONS ##################################
###############################################################################
def cite_lbl(idx):
    if type(idx) == int:
        return df['Author'][idx]+', '+str(int(df['Year'][idx]))
    if type(idx) == list:
        cite_list = []
        c = 1
        for idx_ in idx:
            cite_list.append(str(c)+'. '+df['Author'][idx_]+', '+str(int(df['Year'][idx_])))
            c += 1
        return cite_list
###############################################################################
################# Barplot for Papers/Years (selected articles) ################
###############################################################################
plt.figure(figsize = (5,3.5), dpi =300)
graph = sns.countplot(df['Year'].astype(int), palette = "flare")
graph.set_title("Papers per Year")
graph.set_ylabel("Number of articles")
graph.set_xlabel("Year")
plt.ylim([0,18])
plt.show()

## Barplot for Papers/Years (all articles)
a  = [2020, 2015, 2011, 2017, 2019, 2018, 2018, 2020, 2012, 2020, 2016, 2015, 2017, 2020, 2020, 2017, 2020, 2018, 2014, 2014, 2019, 2014, 2019, 2015, 2016, 2014, 2018, 2015, 2014, 2020, 2020, 2016, 2020, 2020, 2015, 2020, 2019, 2018, 2020, 2020, 2019, 2017, 2020, 2012, 2010, 2016, 2012, 2018, 2015, 2013, 2019, 2016, 2015, 2013, 2016, 2015, 2004, 2019, 2019, 2020, 2018, 2016, 2014, 2013, 2019, 2018, 2018, 2015, 2012, 2019, 2011, 2013, 2018, 2018, 2017, 2015, 2015, 2018, 2016, 2019, 2018, 2014, 2017, 2020, 2019, 2013, 2015, 2017, 2017, 2014, 2018, 2018, 2016, 2015, 2016, 2016, 2017, 2015, 2014, 2018, 2019, 2013, 2013, 2010, 2016, 2019, 2017, 2017, 2016, 2015, 2017, 2015, 2013, 2015, 2018, 2013, 2019, 2013, 2016, 2017, 2015, 2017, 2011, 2011, 2013, 2019, 2017, 2011, 2019, 2020, 2012, 2014, 2020, 2015, 2019, 2015, 2017, 2015, 2010, 2019, 2017, 2017, 2019, 2014, 2018, 2015, 2019, 2015, 2018, 2019, 2020, 2017]
plt.figure(figsize = (5,3.5), dpi =300)
graph = sns.countplot(a, palette = "rocket")
graph.set_title("Papers per Year")
graph.set_ylabel("Number of articles")
graph.set_xlabel("Year")
# plt.ylim([0,18])
plt.show()
###############################################################################
########################### Point plot acc per year ###########################
###############################################################################
plt.figure(figsize = (5,4), dpi =300)
graph = sns.stripplot(x = df['Year'][idx_acc].astype(int), y = df['On_ACC'][idx_acc].astype(float),
              data = df, edgecolor='gray', size=12.5, alpha = .65, palette='Set2')    
# sns.rugplot(data = df, x = df['Year'][idx_acc], y = df['On_ACC'][idx_acc])
graph.set_title("Accuracy per Year")
graph.set_ylabel("Accuracy [%]")
plt.show()
###############################################################################
########################### Plot age range vs. acc ############################
###############################################################################
sns.set_style("whitegrid")
plt.figure(figsize = (5,7), dpi =500)

c = 1
for _range in idx_age_range:
    a = random.random()
    txt =  str(df['Pop_age'][_range])
    val1 = float(txt[0:df['Pop_age'][_range].index(' to ')] )
    val2 = float(txt[df['Pop_age'][_range].index(' to ') + 4:])
    plt.plot([val1, val2], [df['On_ACC'][_range], df['On_ACC'][_range]],
             '-', color = CLR[_range][1].hex_format(), linewidth = 3,
             alpha = 0.6)
    if c%2 != 0:
        plt.annotate(str(c), (val2, df['On_ACC'][_range]),
                     textcoords = "offset points", xytext = (3*(1+(1-a)),-1), ha = 'left',
                     fontsize = 4)
    else:
        plt.annotate(str(c), (val1, df['On_ACC'][_range]),
                     textcoords = "offset points", xytext = (-3*(1+(1-a)),-1), ha = 'right',
                     fontsize = 4)
    c += 1
plt.legend(cite_lbl(idx_age_range), fontsize = 3.9, loc = "lower right")

for _range in idx_age_range:
    txt =  str(df['Pop_age'][_range])
    val1 = float(txt[0:df['Pop_age'][_range].index(' to ')] )
    val2 = float(txt[df['Pop_age'][_range].index(' to ') + 4:])
    
    plt.plot(val1, df['On_ACC'][_range], '>', color = CLR[_range][1].hex_format(),
             markersize = 1)
    plt.plot(val2, df['On_ACC'][_range], '<', color = CLR[_range][1].hex_format(),
             markersize = 1)

for _range in idx_age_single:
    plt.plot(df['Pop_age'][_range], df['On_ACC'][_range], '^',
             color = CLR[_range<<1][1].hex_format(), markersize = 4,
             markeredgecolor = 'k')
    plt.annotate(cite_lbl(_range), (df['Pop_age'][_range], df['On_ACC'][_range]),
                 textcoords = "offset points", xytext = (2,-1), ha = 'left',
                 fontsize = 4)
    
plt.title("Age range vs. Accuracy ")
plt.ylabel("Accuracy")
plt.xlabel("Age")
plt.show()
###############################################################################
###############################################################################
###############################################################################
## Violin plot
## Acc per paper
plt.figure(figsize = (5,6), dpi =300)
sns.lineplot(data = df, x = df['Year'][idx_hom_acc], y = df['On_ACC'][idx_hom_acc])
sns.rugplot(data = df, x = df['Year'][idx_hom_acc], y = df['On_ACC'][idx_hom_acc])
plt.show()

## Acc per paper
plt.figure(figsize = (5,6), dpi =300)
sns.lineplot(data = df, x = df['Year'][idx_het_acc], y = df['On_ACC'][idx_het_acc])
sns.rugplot(data = df, x = df['Year'][idx_het_acc], y = df['On_ACC'][idx_het_acc])
plt.show()

## Single-brain approach    // Multi-brain approach
plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_hom_acc], y = df['On_ACC'][idx_hom_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_hom_acc], split = True, data = df,
               scale="count", inner="stick")
plt.show()

## Extern input             // Multy-physiological
plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_het_acc], y = df['On_ACC'][idx_het_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_het_acc], split = True, data = df,
               scale="width", inner="stick")
plt.show()




## Single-brain approach    // Multi-brain approach
plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_hom_acc], y = df['On_ACC'][idx_hom_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_hom_acc], split = True, data = df)
plt.show()

## Extern input             // Multy-physiological
plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_acc], y = df['On_ACC'][idx_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_acc], col = df['Control'][idx_het_acc],
               split = False, data = df)
plt.show()



## Extern input             // Multy-physiological
plt.figure(figsize = (10,6), dpi =300)
sns.violinplot(x = df['Year'][idx_acc].astype(int), y = df['On_ACC'][idx_acc].astype(float), 
               hue = df['_Div_input'][idx_acc], split = False,
               scale="width", inner="stick", data = df)
plt.show()




plt.figure(figsize = (5,6), dpi =300)
sns.violinplot(x = df['_Div_input'][idx_acc], y = df['On_ACC'][idx_acc].astype(float), 
               split = False, data = df, scale="count", inner="stick")
plt.show()



plt.figure(figsize = (10,6), dpi =300)
sns.violinplot(x = df['Control'][idx_acc], y = df['On_ACC'][idx_acc].astype(float), 
               split = False, data = df, scale="count", inner="stick", palette="flare")
plt.show()

