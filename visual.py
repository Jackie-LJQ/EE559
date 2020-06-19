# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:20:48 2020

@author: liu
"""

import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy import stats

fig, ax = plt.subplots()
df = pd.read_csv('data/features.csv')
#change measure to plot different column
measure = 'zrange'
#df = df[(np.abs(stats.zscore(df[measure])) < 3)].reset_index()
vmin, vmax = int(df[measure].min())-1, int(df[measure].max())+1
bins = range(vmin,vmax,10)

for i in range(1,6):
    df2 = df.loc[df['Class']==i]
    df2 = df2[['Class',measure,'user']]
    df1 = df2.groupby([pd.cut(df2[measure],bins)])['user'].count()
    df1.plot(kind='line',ax=ax,label='class'+str(i))
    plt.ylabel('number of data points')
    plt.legend()
    plt.xticks(rotation=90)

##use bar chart represent 'missing' column
#measure = 'missing'
#fig, ax = plt.subplots()
##df = df[(np.abs(stats.zscore(df[measure])) < 3)].reset_index()
#vmin, vmax = int(df[measure].min())-1, int(df[measure].max())+1
#bins = range(vmin,vmax,1)
##c=['nun','b','g','r','m','k']
#df2 = df.loc[df['Class']==1]
#df2 = df2[['Class',measure,'user']]
#df1 = df2.groupby([pd.cut(df2[measure],bins)])['user'].count().to_frame()
#df1 = df1.rename(columns={'user':'Class1'})
#for i in range(2,6):
#    df2 = df.loc[df['Class']==i]
#    df2 = df2[['Class',measure,'user']]
#    name = 'Class'+str(i)
#    df1[name] = df2.groupby([pd.cut(df2[measure],bins)])['user'].count()
#df1['index'] = range(3,28)
#df1 = df1.set_index('index')
#df1.plot(kind='bar',ax=ax, width=1)
#plt.ylabel('number of data points')
#plt.xlabel('missing')
#plt.legend()
