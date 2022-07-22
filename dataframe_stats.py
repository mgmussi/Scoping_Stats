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
import re
import math


# execfile("ColorMap.py")
exec(open("ColorMap.py").read())
sns.set_style("darkgrid")

cwd = os.getcwd()+'/__Dataframe_ScopingReview.xlsx'
dfa = pd.read_excel(cwd, sheet_name=0, header=[0])
df = pd.read_excel(cwd, sheet_name=1, header=[0])
dfc = pd.read_excel(cwd, sheet_name=2, header=[0])
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
# print(df)
# print(df['ID'].values)
# print(df['On_ACC'].values)

###############################################################################
######################## COUNT NUM OF TAXONOMIC SYSTEMS #######################
###############################################################################
tot_sys = int(sum(df['UN_SYS'].values))

#Lists of rows
names_lbls = ["_Div_input", "_Div_input_sp", "_Mental_str", "_Role_op",
              "_Mode_op", "_Brain_sign", "_Stim_mod", "Control"]
#Empty lists of lists for features names and counted values
names_feat = [[] for _ in range(len(names_lbls))]
total_feat = [[] for _ in range(len(names_lbls))]

for i, name in enumerate(names_lbls):
    #for every row extract data from the column
    #and append to the list of features if is its
    #first appearence
    for _, row in df.iterrows(): 
        # temp = row[name]        
        temp = row[name].split(',')
        for elem in temp:
            if elem != '-':
                if not(any(re.fullmatch(elem, n) for n in names_feat[i])): names_feat[i].append(elem)
        # if not(any(re.fullmatch(temp, n) for n in names_feat[i])): names_feat[i].append(temp)
    #for every feature name, count how many

    for f in names_feat[i]:
        if(name != "Processing_software"):
            total_feat[i].append(int(sum([row['UN_SYS'] for _, row in df.iterrows() if(row[name] == f and int(row['UN_SYS']))])))
        else:
            total_feat[i].append(0)
            for _, row in df.iterrows():
                temp = row[name].split(',')
                for elem in temp:
                    if(elem == f and row['UN_SYS']):
                        total_feat[i][-1] += 1
        
def pt(val):
    return(float(val/tot_sys*100))

def ptp(ls):
    perc = []
    for it in ls:
        perc.append(float(it/sum(ls)*100))
    return perc

def pr_v(vals, ct_str):
    tots = 0
    for val, ct in zip(vals, ct_str):
        print(">> {:2d}: ({:.2f}%) {}".format(val, pt(val), ct))
        tots += val
    # print("<< {}: Sum>>".format(tots))

print("------------\n>> {0}: TOTAL NUMBER OF SYSTEMS\n".format(tot_sys))
for tots, names in zip(total_feat, names_feat):
    pr_v(tots, names)
    print()
###############################################################################
############################ FIND RELEVANT INDEXES ############################
###############################################################################
idx = (df['On_ACC'].values != '-')
idx_include = [counter for counter, value in enumerate(np.array(dfa['Include'])) if value]
idx_un_sys = [counter for counter, value in enumerate(np.array(df['UN_SYS'])) if value]
idx_dif_pop = [counter for counter, value in enumerate(np.array(df['Diff_pop'])) if value]
idx_sys = [counter for counter, value in enumerate(np.array(df['Acq_sys'])) if value != '-']
idx_acc = [counter for counter, value in enumerate(np.array(df['On_ACC'])) if value != '-']
idx_acc2 = [counter for counter, value in enumerate(np.array(dfc['On_ACC'])) if value != '-']
idx_age = [counter for counter, value in enumerate(np.array(df['Pop_age'])) if value != '-']
idx_age_to_ = [counter for counter, value in enumerate(np.array(df['Pop_age'].astype(str))) if 'to' in value]
idx_age_ = [counter for counter, value in enumerate(np.array(df['Pop_age'].astype(str))) if not('to' in value)]
idx_age_sv = [counter for counter, value in enumerate(np.array(df['Pop_age'][idx_age].astype(str))) if not('-' in value)]
idx_DivIn = [[counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Heterogeneous'],
             [counter for counter, value in enumerate(np.array(df['_Div_input'])) if value == 'Homogeneous']]
idx_MeStr = [[counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Selective Attention'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning,Selective Attention']]
idx_MeStr = [[counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Selective Attention'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning'],
             [counter for counter, value in enumerate(np.array(df['_Mental_str'])) if value == 'Operant Conditioning,Selective Attention']]
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
for _, row in df.iterrows():
    if row['UN_SYS']:
        str_array = row['_Brain_sign'].split(',')
        for a, b in itertools.product(str_array, repeat = 2):
            comb_parad[idx[a],idx[b]] += 1
for c,x in enumerate(comb_parad):
    comb_parad[c,5] = (comb_parad[c,c]*2) - sum(comb_parad[c])
comb_parad[3,2] = 0 #since all mu paradigms have SMR,
comb_parad[3,5] = 2 #I'll disconsider them and explain on the paper
print(comb_parad)

##Number of Div of Input
comb_div_input = np.zeros([2,2]).astype(int)
idx_1 = {'Homogeneous':0, 'Heterogeneous':1}
idx_2 = {'Single-Brain Approach':0, 'Multi-Brain Approach':1,
         'Multi-Physiological':2, 'External Input':3}
for _, row in df.iterrows():
    if row['UN_SYS']:
        value = row['_Div_input_sp']
        if(idx_2[value]<2):
            comb_div_input[idx_1['Homogeneous'], idx_2[value]%2] += 1
        else:
            comb_div_input[idx_1['Heterogeneous'], idx_2[value]%2] += 1
print(comb_div_input)
        
##Combination of Simtuli Modalities
comb_stim_mod = np.zeros([4,5]).astype(int)
idx = {'Visual':0, 'Tactile':1, 'Operant':2, 'Auditory':3}
# for value in np.array(df['_Stim_mod']):
for _, row in df.iterrows():
    if row['UN_SYS']:
        str_array = row['_Stim_mod'].split(',')
        for a, b in itertools.product(str_array, repeat = 2):
            comb_stim_mod[idx[a],idx[b]] += 1
for c,x in enumerate(comb_stim_mod):
    comb_stim_mod[c,4] = (comb_stim_mod[c,c]*2) - sum(comb_stim_mod[c])
print(comb_stim_mod)

##Number of Role of Operation
comb_role_op = np.zeros([2]).astype(int)
idx = {'Simultaneous':0, 'Sequential':1}
# for value in np.array(df['_Role_op']):
for _, row in df.iterrows():
    if row['UN_SYS']:
        value = row['_Role_op']
        comb_role_op[idx[value]] += 1
print(comb_role_op)

##Number of Mode of Operation
comb_mode_op = np.zeros([3]).astype(int)
idx = {'Synchronous':0, 'Asynchronous':1, 'Synchronous,Asynchronous':2}
# for value in np.array(df['_Role_op']):
for _, row in df.iterrows():
    if row['UN_SYS']:
        value = row['_Mode_op']
        comb_mode_op[idx[value]] += 1
print(comb_mode_op)
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
################## Barplot for Papers/Years (selected vs all) #################
###############################################################################
year_lbl = []
year_t_all = []
year_t_sel = []
offset = 101
bar_colors = [CLR[x+offset][1].hex_format() for x in list(range(0,13))]
for _, row in dfa.iterrows(): 
    temp = row['Year']        
    if not(any(temp == n for n in year_lbl)): year_lbl.append(temp)
#for every feature name, count how many
for f in year_lbl:
    year_t_sel.append(int(sum([1 for _, row in dfa.iterrows() if(row['Year'] == f and int(row['Include']))])))
    year_t_all.append(int(sum([1 for _, row in dfa.iterrows() if(row['Year'] == f)])))
    
sns.set_style("darkgrid")
# sns.set_style("whitegrid")
plt.figure(figsize = (5,3), dpi =300)
# graph = sns.countplot(x = year_lbl,
#                       y = year_t_all,
#                       palette = "flare",
#                       alpha = 0.5
#                       )
# graph = sns.countplot(x = year_lbl,
#                       y = year_t_sel,
#                       palette = "flare")
plt.bar(year_lbl,
        year_t_all,
        alpha = 0.25,
        color = bar_colors,
        edgecolor = bar_colors,
        )
plt.bar(year_lbl,
        year_t_sel,
        alpha = 1,
        color = bar_colors,
        edgecolor = bar_colors,
        )
plt.title("Papers per Year")
plt.ylabel("Number of articles")
plt.xlabel("Year")
plt.xticks(list(range(2004,2021)))
# plt.ylim([0,18])
# plt.yticks(range(0,18,2))
plt.xticks(rotation=90)
plt.grid(axis="x")
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
###############################################################################
######################## Barplot for Processing systems #######################
###############################################################################
names_soft = []
total_soft = []
for _, row in df.iterrows(): 
    # temp = row[name]        
    temp = row["Processing_software"].split(',')
    for elem in temp:
        if elem != '-':
            if not(any(re.fullmatch(elem, n) for n in names_soft)): names_soft.append(elem)
#for every feature name, count how many
for f in names_soft:
    total_soft.append(0)
    for _, row in df.iterrows():
        temp = row["Processing_software"].split(',')
        for elem in temp:
            if(elem == f and row['UN_SYS']):
                total_soft[-1] += 1
                
total_soft, names_soft = (list(t) for t in zip(*sorted(zip(total_soft, names_soft))))
total_soft.reverse()
names_soft.reverse()
offset = 18
bar_colors = [CLR3[x+offset] for x in list(range(0,len(names_soft)))]
                
sns.set_style("darkgrid")
plt.figure(figsize = (5,3.5), dpi =300)
plt.bar(names_soft,
        total_soft,
        alpha = 1,
        color = bar_colors,
        edgecolor = "white",
        )
plt.title("Development tools")
plt.ylabel("Number of appearances")
plt.xlabel("")
plt.xticks(rotation=90)
plt.grid(axis="x")
plt.xlim([-0.75,len(names_soft)-.25])
# plt.yticks(range(0,18,2))
plt.show()
###############################################################################
######################## Barplot for Acq Sys appearances ######################
###############################################################################
sns.set_style("darkgrid")
plt.figure(figsize = (5,3.5), dpi =300)
graph = sns.countplot(df['Acq_sys'][idx_sys],
                      palette = "mako",
                      order = df['Acq_sys'][idx_sys].value_counts().index)
graph.set_title("Acquisition systems")
graph.set_ylabel("Number of appearances")
graph.set_xlabel("System")
plt.xticks(rotation=90)
# plt.ylim([0,18])
# plt.yticks(range(0,18,2))
plt.show()
###############################################################################
######################## Barplot for Pop Siz appearances ######################
###############################################################################
sns.set_style("darkgrid")
pop = df['Pop_size'][idx_dif_pop].values
avg_pop = np.mean(pop)
std_pop = np.std(pop)
offset = -3
plt.figure(figsize = (5,3.5), dpi =300)
graph = sns.countplot(pop.astype(int),
                      palette = "dark:salmon_r",
                      # order = df['Pop_size'][idx_sys].value_counts().index
                      )
plt.plot([avg_pop+offset, avg_pop+offset], [0, 11], 'r.-', linewidth = .5)
plt.plot([avg_pop-std_pop+offset, avg_pop-std_pop+offset], [0, 11], 'r--', linewidth = .5)
plt.plot([avg_pop+std_pop+offset, avg_pop+std_pop+offset], [0, 11], 'r--', linewidth = .5)
graph.set_title("Population Size")
graph.set_ylabel("Number of appearances")
graph.set_xlabel("Number of participants")
plt.xticks(rotation=0)

# locs, labels = plt.xticks()  # Get the current locations and labels.
# print(locs)
# print(labels)
plt.ylim([0,10.5])
# plt.yticks(range(0,18,2))
plt.show()

print("Average Pop. Size: {:.1f}±{:.1f}".format(avg_pop, std_pop))
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
################## Point plot & boxplot acc per num commands ##################
###############################################################################
sns.set_style("darkgrid")
plt.figure(figsize = (5,3), dpi =300)
graph = sns.boxplot(x = dfc['Num_steps_command_max'][idx_acc2].astype(int),
                      y = dfc['On_ACC'][idx_acc2].astype(float),
                      data = dfc,
                      # edgecolor='gray',
                      # size=12.5,
                      # alpha = .65,
                      palette='Set2')
graph2 = sns.stripplot(x = dfc['Num_steps_command_max'][idx_acc2].astype(int),
                      y = dfc['On_ACC'][idx_acc2].astype(float),
                      data = dfc,
                      marker = 'P',
                      size = 6,
                      # alpha = .65,
                      edgecolor = 'gray',
                      linewidth = .5,
                      palette='Set2',
                      color = 'gray')
# sns.rugplot(data = df, x = df['Year'][idx_acc], y = df['On_ACC'][idx_acc])
graph.set_title("Accuracy per Number of Commands")
graph.set_ylabel("Accuracy [%]")
graph.set_xlabel("Maximum number of commands")
plt.show()
###############################################################################
################ Point plot & boxplot acc per individual input ################
###############################################################################
Ins_lbl = []
Ins_val = []
sns.set_style("darkgrid")
plt.figure(figsize = (4,3), dpi =300)
for _, row in df.iterrows():
    if row['In1_on_class'] != '-' and row['In1_on_class'] != '--' and not(math.isnan(row['In1_on_class'])):
        Ins_lbl.append(row['In1'])
        Ins_val.append(row['In1_on_class'])
    if row['In2_on_class'] != '-' and row['In2_on_class'] != '--' and not(math.isnan(row['In2_on_class'])):
        Ins_lbl.append(row['In2'])
        Ins_val.append(row['In2_on_class'])
# In1 = [list(df['In1']), list(df['In1_on_class'])]
# In2 = [list(df['In2']), list(df['In2_on_class'])]
##append lists together and then plot acc for each type of paradigm
graph = sns.boxplot(x = Ins_lbl,
                    y = Ins_val,
                    data = dfc,
                    color='gray',
                    boxprops=dict(alpha=1),
                    linewidth = 1,
                    palette='Set3')
g = sns.stripplot(x = Ins_lbl,
                  y = Ins_val,
                  alpha = .9,
                  marker = 'P',
                  size = 5,
                  # alpha = .65,
                  edgecolor = 'black',
                  linewidth = .5,
                  palette= 'Set3')    
g.set_title("Accuracy per individual input type")
g.set_ylabel("Accuracy [%]")
g.set_xlabel("")
plt.xticks(rotation=90)
plt.show()
###############################################################################
######################### Box plot acc per Role of Op #########################
###############################################################################
sns.set_style("darkgrid")
plt.figure(figsize = (5,4), dpi =300)
# graph = sns.stripplot(x = df['_Role_op'][idx_acc], y = df['On_ACC'][idx_acc].astype(float),
#               data = df, edgecolor='gray', size=12.5, alpha = .65, palette='Set2')    
graph = sns.boxplot(x = df['_Role_op'][idx_acc],
                    y = df['On_ACC'][idx_acc].astype(float),
                    data = df,
                    # edgecolor='gray',
                    # size=12.5,
                    # alpha = .65,
                    palette='Set2')   
# sns.rugplot(data = df, x = df['Year'][idx_acc], y = df['On_ACC'][idx_acc])
graph.set_title("Accuracy per Role of Operation")
graph.set_ylabel("Accuracy [%]")
graph.set_xlabel("Role of Operation")
graph.set_xticklabels(["Sequential", "Simultaneous"])
plt.show()
###############################################################################
######################### Box plot acc per Stim Modal #########################
###############################################################################
sns.set_style("darkgrid")
plt.figure(figsize = (10,4), dpi =300)
# graph = sns.stripplot(x = df['_Role_op'][idx_acc], y = df['On_ACC'][idx_acc].astype(float),
#               data = df, edgecolor='gray', size=12.5, alpha = .65, palette='Set2')    
graph = sns.boxplot(x = df['_Stim_mod'][idx_acc],
                    y = df['On_ACC'][idx_acc].astype(float),
                    data = df,
                    order = ["Visual", "Visual,Auditory", "Visual,Operant", 
                             "Visual,Operant,Auditory", "Operant",
                             "Auditory", "Operant,Tactile", "Tactile"],
                    # edgecolor='gray',
                    # size=12.5,
                    # alpha = .65,
                    palette='Set2')   
# sns.rugplot(data = df, x = df['Year'][idx_acc], y = df['On_ACC'][idx_acc])
graph.set_title("Accuracy per Stimulus Modalities")
graph.set_ylabel("Accuracy [%]")
graph.set_xlabel("\nRole of Operation")
graph.set_xticklabels(["Visual", "Visual &\nAuditory", "Visual &\nOperant",
                       "Visual,\nOperant &\nAuditory", "Operant", "Auditory",
                       "Operant &\nTactile", "Tactile"])
plt.tick_params(size = 1, labelsize = 9)
plt.show()


sns.set_style("darkgrid")
plt.figure(figsize = (10,4), dpi =300)

graph = sns.boxplot(x = df['Num_commands'][idx_acc],
                    y = df['On_ACC'][idx_acc].astype(float),
                    data = df,
                    # order = ["Visual", "Visual,Auditory", "Visual,Operant", 
                    #          "Visual,Operant,Auditory", "Operant",
                    #          "Auditory", "Operant,Tactile", "Tactile"],
                    # edgecolor='gray',
                    # size=12.5,
                    # alpha = .65,
                    palette='Set2')   
# sns.rugplot(data = df, x = df['Year'][idx_acc], y = df['On_ACC'][idx_acc])
graph.set_title("Accuracy per Number of Targets")
graph.set_ylabel("Accuracy [%]")
graph.set_xlabel("\nNumber of Targets")
# graph.set_xticklabels(["Visual", "Visual &\nAuditory", "Visual &\nOperant",
#                        "Visual,\nOperant &\nAuditory", "Operant", "Auditory",
#                        "Operant &\nTactile", "Tactile"])
plt.tick_params(size = 1, labelsize = 9)
plt.show()
###############################################################################
########################### Plot age range vs. acc ############################
###############################################################################
'''
Perhaps we only need two or three colours:
    those with only neurotypical people,
    those with both people with and without disabilities,
    and those with only people with disabilities.
'''
sns.set_style("whitegrid")
plt.figure(figsize = (5,7), dpi =500)
c = 1
he = 35     #blue = healthy only
nh = 25     #orange = disability only
bo = 100    #green = both

SEPARATE = True


def sep_colors(r):
    author = df['Author'][r]
    cond = df['Pop_cond'][r]
    if(author == 'Nann et al.' or author == 'Brennan et al.' or author == 'Soekadar et al.'):
        if cond != 'Healthy':
            the_color = CLR[bo][1].hex_format()
        else:
            the_color = CLR[nh][1].hex_format()
    else:
        the_color = CLR[he][1].hex_format()
    return the_color



for _range in idx_age_range:
    a = 1.1*random.random()
    txt =  str(df['Pop_age'][_range])
    val1 = float(txt[0:df['Pop_age'][_range].index(' to ')] )
    val2 = float(txt[df['Pop_age'][_range].index(' to ') + 4:])
    
    if SEPARATE:
        ##To separate by color
        the_color = sep_colors(_range)
    else:
        ##To keep multi-color
        the_color = CLR[_range][1].hex_format()
        
    plt.plot([val1, val2],
             [df['On_ACC'][_range],
              df['On_ACC'][_range]],
             '-',
             color = the_color,
             linewidth = 3,
             alpha = 0.6)
    if c%2 != 1:
        plt.annotate(str(c),
                     (val2, df['On_ACC'][_range]),
                     textcoords = "offset points",
                     xytext = (3.5*(1+(1-a)),-1),
                     ha = 'left',
                     fontsize = 4)
    # elif c%4 == 1:
    #     plt.annotate(str(c),
    #                  (val1, df['On_ACC'][_range]),
    #                  textcoords = "offset points",
    #                  xytext = (-5*(1+(1-a)),-1),
    #                  ha = 'right',
    #                  fontsize = 4)
    else:
        plt.annotate(str(c),
                     (val1, df['On_ACC'][_range]),
                     textcoords = "offset points",
                     xytext = (-3.5*(1+(1-a)),-1),
                     ha = 'right',
                     fontsize = 4)
    c += 1
plt.legend(cite_lbl(idx_age_range), fontsize = 3.9, loc = "lower right")

for _range in idx_age_range:
    txt =  str(df['Pop_age'][_range])
    val1 = float(txt[0:df['Pop_age'][_range].index(' to ')] )
    val2 = float(txt[df['Pop_age'][_range].index(' to ') + 4:])
    
    if SEPARATE:
        ##To separate by color
        the_color = sep_colors(_range)
    else:
        ##To keep multi-color
        the_color = CLR[_range][1].hex_format()
    
    plt.plot(val1,
             df['On_ACC'][_range],
             '>',
             color = the_color,
             markersize = 1)
    plt.plot(val2,
             df['On_ACC'][_range],
             '<',
             color = the_color,
             markersize = 1)

for _range in idx_age_single:
    plt.plot(df['Pop_age'][_range], df['On_ACC'][_range], '^',
             color = CLR[_range<<1][1].hex_format(), markersize = 4,
             markeredgecolor = 'k')
    plt.annotate(cite_lbl(_range), (df['Pop_age'][_range], df['On_ACC'][_range]),
                 textcoords = "offset points", xytext = (2,-1), ha = 'left',
                 fontsize = 4)
    
plt.title("Age range vs. Accuracy", fontsize = 8)
plt.ylabel("Accuracy [%]", fontsize = 6)
plt.xlabel("Age [years]", fontsize = 6)
# plt.yscale("log")
plt.tick_params(size = 1, labelsize = 6)
plt.show()
#/**
# * Age ranges overlap
# */
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = (8,1.2), dpi =500)
c = 1
for _range in idx_age_range:
    a = 1.1*random.random()
    txt =  str(df['Pop_age'][_range])
    val1 = float(txt[0:df['Pop_age'][_range].index(' to ')] )
    val2 = float(txt[df['Pop_age'][_range].index(' to ') + 4:])
    plt.plot([val1, val2], [1, 1],
             '-',
             color = 'k',
             linewidth = 15,
             alpha = 0.1)
plt.title("Age ranges overlap", fontsize = 10)
plt.ylabel("", fontsize = 6)
plt.xlabel("Age [years]", fontsize = 9)
# plt.yscale("log")
plt.tick_params(size = 1, labelsize = 9)
plt.grid()
# plt.axis("off")
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    # labelbottom=False, # labels along the bottom edge are off
    left=False,
    right=False,
    labelleft=False)
plt.ylim(0.999,1.001)
plt.tight_layout()
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
sns.set_style("whitegrid")
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
sns.set_style("whitegrid")
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

offset = 85
main_lbl   = ['Visual', 'Tactile', 'Operant', 'Auditory']
main_sizes = [data[i,i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))


sub_size = [data[a,b] for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_colors = [CLR[b+offset][1].hex_format() if b<lnn else CLR[a+offset][1].hex_format() for a,b in itertools.product(range(ln), repeat = 2) if a!=b and a<lnn]
sub_explode = 0*np.ones(len(sub_size))

# Plot
sns.set_style("whitegrid")
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
#/**
# * Role of Operation
# */
main_lbl = []
main_sizes = []
main_colors = []
data = comb_role_op
ln = len(data)
lnn = ln - 1

offset = 100
main_lbl   = ['Sequential', 'Simultaneous']
main_sizes = [data[i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))


sub_size = []
sub_colors = []
sub_explode = 0*np.ones(len(sub_size))

# Plot
sns.set_style("whitegrid")
plt.figure(figsize = (4,4), dpi =500)
plt.title("Percentage of Roles of Operation\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = -65,
        frame = True,
        textprops={"fontsize": 8},
        explode = main_explode)
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = 0,
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
# * Mode of Operation
# */
main_lbl = []
main_sizes = []
main_colors = []
data = comb_mode_op
ln = len(data)
lnn = ln - 1

offset = 55
main_lbl   = ['Synchronous', 'Asynchronous', 'Synchronous &\nAsynchronous']
main_sizes = [data[i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))


sub_size = []
sub_colors = []
sub_explode = 0*np.ones(len(sub_size))

# Plot
sns.set_style("whitegrid")
plt.figure(figsize = (4,4), dpi =500)
plt.title("Percentage of Modes of Operation\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = 110,
        frame = True,
        textprops={"fontsize": 8},
        explode = main_explode)
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = 0,
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
# * Control
# */
main_lbl = []
main_sizes = []
main_colors = []
total_feat[7].sort()
data = total_feat[7]
ln = len(data)
lnn = ln - 1

offset = 100
main_lbl   = ['Home automation',
              'Cursor/\nGame',
              'Robot/\nRobotic Hand',
              'Drone/Vehicle/\nWheelchair',
              'Speller',
              'Unspecified\nDevice']
main_sizes = [data[i] for i in range(len(data))]
main_colors= [CLR[i+offset][1].hex_format() for i in range(len(data))]
main_explode = 0*np.ones(len(main_lbl))
main_perc_lbl = []
for c, elem in enumerate(main_sizes):
    main_perc_lbl.append(main_lbl[c] + ": " + "{:.2f}%".format((elem/sum(main_sizes)*100)))


sub_size = []
sub_colors = []
sub_explode = 0*np.ones(len(sub_size))

# Plot
sns.set_style("whitegrid")
plt.figure(figsize = (4,4), dpi =500)
plt.title("What was being controlled?\n", loc = 'center')
plt.pie(main_sizes,
        labels = main_lbl,
        colors = main_colors,
        startangle = 95,
        frame = True,
        textprops={"fontsize": 8},
        explode = main_explode)
plt.pie(sub_size,
        colors = sub_colors,
        radius = 0.75,
        startangle = 0,
        explode = sub_explode)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()
plt.legend(main_perc_lbl,
          title="Legend",
          loc="lower right",
          bbox_to_anchor=(1, 0, .9, 0))

plt.show()








