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


execfile("ColorMap.py")
CLR = return_palette_list(colors)

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
idx_MeStr = [[counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Selective Attention'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning & Selective Attention']]


idx_het_acc = list(set.intersection(set(idx_DivIn[0]), set(idx_acc)))
idx_hom_acc = list(set.intersection(set(idx_DivIn[1]), set(idx_acc)))
idx_age_single = list(set.intersection(set(idx_age), set(idx_age_), set(idx_acc)))
idx_age_range = list(set.intersection(set(idx_age), set(idx_age_to_), set(idx_acc)))
idx_sel_acc = list(set.intersection(set(idx_MeStr[0]), set(idx_acc)))
idx_opr_acc = list(set.intersection(set(idx_MeStr[1]), set(idx_acc)))
idx_sel_opr_acc = list(set.intersection(set(idx_MeStr[2]), set(idx_acc)))
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
sns.set_style("darkgrid")
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
sns.set_style("darkgrid")
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
    
plt.title("Age range vs. Accuracy")
plt.ylabel("Accuracy [%]")
plt.xlabel("Age [years]")
plt.show()
###############################################################################
################################# Violin plot #################################
###############################################################################
#/**
# * Diversity of Input Comparison
# */
sns.set_style("darkgrid")
a = round(len(CLR)*random.random())
b = round(len(CLR)*random.random())
c = round(len(CLR)*random.random())
d = round(len(CLR)*random.random())
## Single-brain approach // Multi-brain approach
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,7), dpi =300)
fig.suptitle("Accuracy distribution for Diversity of Input", size = 18)
g = sns.violinplot(ax = ax1, 
               x = df['_Div_input'][idx_hom_acc],
               y = df['On_ACC'][idx_hom_acc].astype(float),
               hue = df['_Div_input_sp'][idx_hom_acc],
               split = True,
               data = df,
               scale="count",
               inner="stick",
               palette = [CLR[a][1].hex_format(), CLR[b][1].hex_format()],
               title = '')
g.legend(loc = "lower right", title = "Approach");
ax1.set_ylabel("Accuracy [%]")
ax1.set_xlabel("")
g.set_ylim([30,120])
## Extern input // Multy-physiological
g = sns.violinplot(ax = ax2,
               x = df['_Div_input'][idx_het_acc],
               y = df['On_ACC'][idx_het_acc].astype(float), 
               hue = df['_Div_input_sp'][idx_het_acc],
               split = True,
               data = df,
               scale="width",
               inner="stick",
               palette = [CLR[c][1].hex_format(), CLR[d][1].hex_format()],
               title = '')
g.legend(loc = "lower right", title = "Approach");
ax2.set_ylabel("")
ax2.set_xlabel("")
g.set_ylim([30,120])
plt.show()
#/**
# * Diversity of Mental Strategy
# */
sns.set_style("darkgrid")
a = round(len(CLR)*random.random())
b = round(len(CLR)*random.random())
c = round(len(CLR)*random.random())
plt.figure(figsize = (8,4), dpi =500)
g = sns.violinplot(x = df['_Mental_str'][idx_acc],
               y = df['On_ACC'][idx_acc].astype(float), 
               # hue = df['_Div_input_sp'][idx_sel_acc],
               inner='stick',
               split = False,
               palette = [CLR[a][1].hex_format(),CLR[b][1].hex_format(),CLR[c][1].hex_format()],
               data = df)
g.set_xticklabels(['Operant\nConditioning', 'Operant Conditioning\n& Selective Attention',
                   'Selective\nAttention'])
# sns.set(font_scale = 1)
plt.title("Accuracy distribution for Mental Strategies")
plt.ylabel("Accuracy [%]")
plt.xlabel("")
plt.show()
#/**
# * Diversity of Mental Strategy
# */
sns.set_style("darkgrid")
plt.figure(figsize = (5,4), dpi =300)
g = sns.stripplot(x = df['_Mental_str'][idx_acc],
                      y = df['On_ACC'][idx_acc].astype(float),
                      data = df,
                      edgecolor='white',
                      size=10,
                      alpha = .65,
                      palette=[CLR[a][1].hex_format(),CLR[b][1].hex_format(),CLR[c][1].hex_format()])    
g.set_title("Accuracy distribution for Mental Strategies")
g.set_xticklabels(['Operant\nConditioning', 'Operant Conditioning\n& Selective Attention',
                   'Selective\nAttention'])
g.set_ylabel("Accuracy [%]")
g.set_xlabel("")
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

