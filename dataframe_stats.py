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
import itertools


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

###############################################################################
######################## COUNT NUM OF TAXONOMIC SYSTEMS #######################
###############################################################################
tot_sys = int(sum(df['UN_SYS'].values))
tot_div_in = [int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input'] == 'Heterogeneous' and int(row['UN_SYS']))])),
              int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input'] == 'Homogeneous' and int(row['UN_SYS']))]))]
tot_div_ins = [int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input_sp'] == 'External Input' and int(row['UN_SYS']))])),
               int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input_sp'] == 'Multi-Brain Approach' and int(row['UN_SYS']))])),
               int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input_sp'] == 'Multi-Physiological' and int(row['UN_SYS']))])),
               int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Div_input_sp'] == 'Single-Brain Approach' and int(row['UN_SYS']))]))]
tot_men_str = [int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Mental_str'] == 'Selective Attention' and int(row['UN_SYS']))])),
               int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Mental_str'] == 'Operant Conditioning' and int(row['UN_SYS']))])),
               int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Mental_str'] == 'Operant Conditioning,Selective Attention' and int(row['UN_SYS']))]))]
tot_rol_op = [int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Role_op'] == 'Sequential' and int(row['UN_SYS']))])),
              int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Role_op'] == 'Simultaneous' and int(row['UN_SYS']))]))]
tot_mod_op = [int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Mode_op'] == 'Synchronous' and int(row['UN_SYS']))])),
              int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Mode_op'] == 'Asynchronous' and int(row['UN_SYS']))]))]
b_s = []
s_m = []
for _, row in df.iterrows():
    temp1 = row['_Brain_sign']
    temp2 = row['_Stim_mod']
    if not(any(temp1 in b for b in b_s)): b_s.append(temp1)
    if not(any(temp2 in s for s in s_m)): s_m.append(temp2)
    
tot_br_mode = []
tot_st_mode = []
for brain, stim in zip(b_s, s_m):
    tot_br_mode.append(int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Brain_sign'] == brain and int(row['UN_SYS']))])))
    tot_st_mode.append(int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row['_Stim_mod'] == stim and int(row['UN_SYS']))])))
###############################################################################
############################ FIND RELEVANT INDEXES ############################
###############################################################################
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
############################ PIE PLOT COMBINATIONS ############################
###############################################################################
##Combination of Brain Signals
comb_parad = np.zeros([5,6]).astype(int)
idx = {'SSEP':0, 'ERP':1, 'SMR':2, 'mu':3, 'SCP':4}
for value in np.array(df['_Brain_sign']):
    str_array = value.split(',')
    for a, b in itertools.product(str_array, repeat = 2):
        comb_parad[idx[a],idx[b]] += 1
for c,x in enumerate(comb_parad):
    comb_parad[c,5] = (comb_parad[c,c]*2) - sum(comb_parad[c])
comb_parad[3,2] = 0 #since all mu paradigms have SMR,
comb_parad[3,5] = 1 #I'll disconsider them and explain on the paper
print(comb_parad)

##Number of Div of Input
comb_div_input = np.zeros([2,2]).astype(int)
idx_1 = {'Homogeneous':0, 'Heterogeneous':1}
idx_2 = {'Single-Brain Approach':0, 'Multi-Brain Approach':1,
         'Multi-Physiological':2, 'External Input':3}
for value in np.array(df['_Div_input_sp']):
    if(idx_2[value]<2):
        comb_div_input[idx_1['Homogeneous'], idx_2[value]%2] += 1
    else:
        comb_div_input[idx_1['Heterogeneous'], idx_2[value]%2] += 1
print(comb_div_input)
        
##Combination of Simtuli Modalities
comb_stim_mod = np.zeros([4,5]).astype(int)
idx = {'Visual':0, 'Tactile':1, 'Operant':2, 'Auditory':3}
for value in np.array(df['_Stim_mod']):
    str_array = value.split(',')
    for a, b in itertools.product(str_array, repeat = 2):
        comb_stim_mod[idx[a],idx[b]] += 1
for c,x in enumerate(comb_stim_mod):
    comb_stim_mod[c,4] = (comb_stim_mod[c,c]*2) - sum(comb_stim_mod[c])
# comb_parad[3,2] = 0 #since all mu paradigms have SMR,
# comb_parad[3,5] = 1 #I'll disconsider them and explain on the paper
print(comb_stim_mod)

##Number of Role of Operation
comb_role_op = np.zeros([3]).astype(int)
idx = {'Simultaneous':0, 'Sequential':1, 'Simultaneous,Sequential':2}
for value in np.array(df['_Role_op']):
    comb_role_op[idx[value]] += 1
print(comb_role_op)
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
plt.yticks(range(0,18,2))
plt.show()

## Barplot for Papers/Years (all articles)
# a  = [2020, 2015, 2011, 2017, 2019, 2018, 2018, 2020, 2012, 2020, 2016, 2015, 2017, 2020, 2020, 2017, 2020, 2018, 2014, 2014, 2019, 2014, 2019, 2015, 2016, 2014, 2018, 2015, 2014, 2020, 2020, 2016, 2020, 2020, 2015, 2020, 2019, 2018, 2020, 2020, 2019, 2017, 2020, 2012, 2010, 2016, 2012, 2018, 2015, 2013, 2019, 2016, 2015, 2013, 2016, 2015, 2004, 2019, 2019, 2020, 2018, 2016, 2014, 2013, 2019, 2018, 2018, 2015, 2012, 2019, 2011, 2013, 2018, 2018, 2017, 2015, 2015, 2018, 2016, 2019, 2018, 2014, 2017, 2020, 2019, 2013, 2015, 2017, 2017, 2014, 2018, 2018, 2016, 2015, 2016, 2016, 2017, 2015, 2014, 2018, 2019, 2013, 2013, 2010, 2016, 2019, 2017, 2017, 2016, 2015, 2017, 2015, 2013, 2015, 2018, 2013, 2019, 2013, 2016, 2017, 2015, 2017, 2011, 2011, 2013, 2019, 2017, 2011, 2019, 2020, 2012, 2014, 2020, 2015, 2019, 2015, 2017, 2015, 2010, 2019, 2017, 2017, 2019, 2014, 2018, 2015, 2019, 2015, 2018, 2019, 2020, 2017]
# plt.figure(figsize = (5,3.5), dpi =300)
# graph = sns.countplot(a, palette = "rocket")
# graph.set_title("Papers per Year")
# graph.set_ylabel("Number of articles")
# graph.set_xlabel("Year")
# # plt.ylim([0,18])
# plt.show()
'''
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
'''
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
'''
###############################################################################
################################# Points plot #################################
###############################################################################
#/**
# * Diversity of Mental Strategy Points
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
#/**
# * Diversity of Brain Signals Points
# */
sns.set_style("darkgrid")
a = round(len(CLR)*random.random())
b = round(len(CLR)*random.random())
c = round(len(CLR)*random.random())
d = round(len(CLR)*random.random())
e = round(len(CLR)*random.random())
f = round(len(CLR)*random.random())
g = round(len(CLR)*random.random())
h = round(len(CLR)*random.random())
i = round(len(CLR)*random.random())
j = round(len(CLR)*random.random())
plt.figure(figsize = (8,4), dpi =500)
g = sns.stripplot(x = df['_Brain_sign'][idx_acc],
                      y = df['On_ACC'][idx_acc].astype(float),
                      data = df,
                      edgecolor='white',
                      size=10,
                      alpha = .65,
                      palette=[CLR[a][1].hex_format(),CLR[b][1].hex_format(),
                               CLR[c][1].hex_format(),CLR[d][1].hex_format(),
                               CLR[e][1].hex_format(),CLR[f][1].hex_format(),
                               CLR[g][1].hex_format(),CLR[h][1].hex_format(),
                               CLR[i][1].hex_format(),CLR[j][1].hex_format(),])  
# g.set_xticklabels(['Operant\nConditioning', 'Operant Conditioning\n& Selective Attention',
#                    'Selective\nAttention'])
# sns.set(font_scale = 1)
plt.title("Accuracy distribution for Brain Signals")
plt.ylabel("Accuracy [%]")
plt.xlabel("")
plt.show()
#/**
# * Diversity of Role of Operation points
# */
sns.set_style("darkgrid")
a = round(len(CLR)*random.random())
b = round(len(CLR)*random.random())
c = round(len(CLR)*random.random())
d = round(len(CLR)*random.random())
e = round(len(CLR)*random.random())
f = round(len(CLR)*random.random())
g = round(len(CLR)*random.random())
h = round(len(CLR)*random.random())
i = round(len(CLR)*random.random())
j = round(len(CLR)*random.random())
plt.figure(figsize = (8,4), dpi =500)
g = sns.stripplot(x = df['_Role_op'][idx_acc],
                      y = df['On_ACC'][idx_acc].astype(float),
                      data = df,
                      edgecolor='white',
                      size=10,
                      alpha = .65,
                      palette=[CLR[a][1].hex_format(),CLR[b][1].hex_format(),
                               CLR[c][1].hex_format(),CLR[d][1].hex_format(),
                               CLR[e][1].hex_format(),CLR[f][1].hex_format(),
                               CLR[g][1].hex_format(),CLR[h][1].hex_format(),
                               CLR[i][1].hex_format(),CLR[j][1].hex_format(),])  
# g.set_xticklabels(['Operant\nConditioning', 'Operant Conditioning\n& Selective Attention',
#                    'Selective\nAttention'])
# sns.set(font_scale = 1)
plt.title("Accuracy distribution for Role of Operation")
plt.ylabel("Accuracy [%]")
plt.xlabel("")
plt.show()
'''
###############################################################################
################################## Pie plot ###################################
############################################################################### 
#/**
# * Brain signal
# */
main_lbl = []
main_sizes = []
main_colors = []
data = comb_parad
ln = len(data[0])
lnn = ln - 1


offset = 10
main_lbl   = ['SSEP', 'ERP', 'SMR', 'µ-rhythm', 'SCP']
main_sizes = [data[i,i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))

sub_lbl = ['ERP', 'SMR', 'µ-rhythm', 'SCP', 'self',    #SSEP
           'SSEP', 'SMR', 'µ-rhythm', 'SCP', 'self',    #ERP
           'SSEP', 'ERP', 'µ-rhythm', 'SCP', 'self',    #SMR
           'SSEP', 'ERP', 'SMR', 'SCP', 'self',         #µ-rhythm
           'SSEP', 'ERP', 'SMR', 'µ-rhythm', 'self']    #SCP

sub_size = [data[a,b] for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_colors = [CLR[b+offset][1].hex_format() if b<lnn else CLR[a+offset][1].hex_format() for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_explode = 0*np.ones(len(sub_size))

# Plot
plt.figure(figsize = (4,4), dpi =500)
plt.title("Percentage of Paradigms\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = 90,
        frame = True,
        textprops={"fontsize": 8},
        explode = main_explode)
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = 90,
        explode = sub_explode)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.legend(main_perc_lbl,
          title="Legend",
          loc="lower left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()
#/**
# * Diversity of Input
# */
main_lbl = []
main_sizes = []
main_colors = []
data = comb_div_input
ln = len(data[0])
lnn = ln - 1

offset = 50
main_lbl   = ['Heterogeneous', 'Homogeneous']
main_sizes = [sum(x) for x in data]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))

sub_lbl = ['Single-Brain Approach', 'Multi-Brain Approach',
         'Multi-Physiological', 'External Input']
sub_size = data.reshape(4).tolist()
sub_colors = [CLR[2+a+offset][1].hex_format() for a in range(len(sub_size))]
for c, elem in enumerate(sub_size):
    main_perc_lbl.append(sub_lbl[c] + ": " + "{:.2f}%".format((elem/main_sizes[int(np.floor(c/2))])*100))

# Plot
plt.figure(figsize = (4,4), dpi =500)
plt.title("Percentage of Diversity of Input\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = -45,
        frame = True,
        textprops={"fontsize": 8})
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = -45,
        labeldistance = 0.5)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.legend(main_perc_lbl,
          title="Legend",
          loc="lower left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()
#/**
# * Stimulus modalities
# */
main_lbl = []
main_sizes = []
main_colors = []
data = comb_stim_mod
ln = len(data[0])
lnn = ln - 1

offset = 10
main_lbl   = ['Visual', 'Tactile', 'Operant', 'Auditory']
main_sizes = [data[i,i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))

sub_lbl = ['ERP', 'SMR', 'µ-rhythm', 'SCP', 'self',    #SSEP
           'SSEP', 'SMR', 'µ-rhythm', 'SCP', 'self',    #ERP
           'SSEP', 'ERP', 'µ-rhythm', 'SCP', 'self',    #SMR
           'SSEP', 'ERP', 'SMR', 'SCP', 'self',         #µ-rhythm
           'SSEP', 'ERP', 'SMR', 'µ-rhythm', 'self']    #SCP

sub_size = [data[a,b] for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_colors = [CLR[b+offset][1].hex_format() if b<lnn else CLR[a+offset][1].hex_format() for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_explode = 0*np.ones(len(sub_size))

# Plot
plt.figure(figsize = (4,4), dpi =500)
plt.title("Percentage of Stimulus Modalities\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = 90,
        frame = True,
        textprops={"fontsize": 8},
        explode = main_explode)
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = 90,
        explode = sub_explode)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.legend(main_perc_lbl,
          title="Legend",
          loc="lower left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()