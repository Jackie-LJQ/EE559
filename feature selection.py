## -*- coding: utf-8 -*-
#"""
#Created on Tue Apr 28 21:10:55 2020
#
#@author: liu
#"""
#

from utils import Config
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn import feature_selection
import seaborn as sn
import matplotlib.pyplot as plt
#

#variance of each feature use centred but not scaled data
df = pd.read_csv(Config['feature_path'])
#var = []
#columns = ['missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
#       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax']
#for i in columns:
#    j = df[[i]].to_numpy()
#    print(i,np.var(j))

# PCA
feature = df[['missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax']]
#feature = feature.to_numpy()
##V = np.cov(feature)
V = feature.corr()
valves, vector = eig(V)

#the covariance matrix of features and labels
#feature = df[['Class','missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
#       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax']]
#corr_matrix = feature.corr()
#sn.heatmap(corr_matrix, annot=True,cmap='Blues', fmt='.2g')
#plt.show()

#