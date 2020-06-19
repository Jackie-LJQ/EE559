# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:06:52 2020

@author: liu
"""

from sklearn.linear_model import Perceptron, RidgeClassifier, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import sklearn.naive_bayes as sk
from sklearn.neighbors import NearestCentroid

Config ={}
# you should replace it with your own root_path
#Config['feature_path'] = 'data/normalize features.csv'
#Config['feature_path'] = 'data/features.csv'
Config['feature_path'] = 'data/standardize features.csv'
#Config['test_path'] = 'data/normalize test features.csv'
Config['test_path'] = 'data/standardize test features.csv'
#Config['test_path'] = 'data/test features.csv'


Config['debug'] = False

#Config['selected_features'] =['Class', 'user', 'missing']

Config['selected_features'] = ['Class', 'user', 'missing',\
      'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
       'ydev', 'ymin', 'ymax', 'zmean', 'zdev',\
       'zmin', 'zmax','xrange', 'yrange', 'zrange']

#Config['selected_features'] = ['Class', 'user', 'ydev', 'zrange', 'zmax','ymax', 'missing']

Config['model']= Perceptron()
#Config['model'] = SVC(gamma='auto')
#Config['model']=SVC(kernel='linear')
#Config['model'] = GaussianNB()
#Config['model'] = RidgeClassifier()
#Config['model'] = NearestCentroid()
#Config['model'] = LinearRegression()