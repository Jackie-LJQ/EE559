## -*- coding: utf-8 -*-
#"""
#Created on Mon Apr 20 11:19:49 2020
#
#@author: liu
#"""
## extract features from data set
## for both training and testing data set
import pandas as pd
##
##df=pd.read_csv('data/D_train.csv')
#df = pd.read_csv('data/D_test.csv')
###number of missing value
#features = pd.DataFrame(df['Class'])
#features['user'] = df['User']
#features['missing'] = pd.DataFrame(df.isnull().sum(axis=1)/3)
#
#names = ['X'+str(i) for i in range(12)]
#features['xmean'] = df[names].mean(axis=1)
#features['xdev'] = df[names].std(axis=1)
#features['xmin'] = df[names].min(axis=1)
#features['xmax'] = df[names].max(axis=1)
###
#names = ['Y'+str(i) for i in range(12)]
#features['ymean'] = df[names].mean(axis=1)
#features['ydev'] = df[names].std(axis=1)
#features['ymin'] = df[names].min(axis=1)
#features['ymax'] = df[names].max(axis=1)
####
#names = ['Z'+str(i) for i in range(12)]
#features['zmean'] = df[names].mean(axis=1)
#features['zdev'] = df[names].std(axis=1)
#features['zmin'] = df[names].min(axis=1)
#features['zmax'] = df[names].max(axis=1)
##
#
#df1 = features[['user', 'xmin', 'xmax','ymin', 'ymax', 'zmin', 'zmax']]
#for i in ['x','y','z']:   
#    Range, Max, Min = i+'range', i+'max', i+'min'
#    df1[Range] = df1[Max]-df1[Min]
##    mean = df1[Range].groupby(df1.user).mean().reset_index()
##    df1 = df1.merge(mean, on='user')
##    rangex, rangey = Range+'_x', Range+'_y'
#    features[Range] = pd.DataFrame(df1[Range])


#features.to_csv('data/features.csv', index=False)
#features.to_csv('data/test features.csv', index=False)
#
#
## normalized training features
#df = pd.read_csv('data/features.csv')
#df1 = pd.DataFrame(df[['Class','user']])
###
#columns = ['missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
#       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax','xrange', 'yrange', 'zrange']
## scale the range of features to 0-1, don't change variance
##
#for name in columns:
#    minimum = df[name].min()
#    maximum = df[name].max()
#    df1[name] = (df[name]-minimum)/(maximum-minimum)

#df1.to_csv('data/normalize features.csv', index=False)


### standardized training features
#df = pd.read_csv('data/features.csv')
#df1 = pd.DataFrame(df[['Class','user']])
#columns = ['missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
#       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax','xrange', 'yrange', 'zrange']
#for name in columns:
#    meanvalue = df[name].mean()
#    stdev = df[name].std()
#    df1[name] = (df[name]-meanvalue)/stdev
#
#df1.to_csv('data/standardize features.csv', index=False)


#normalize and standardize testing data
df = pd.read_csv('data/test features.csv')
df1 = pd.read_csv('data/features.csv')

columns = ['missing', 'xmean', 'xdev', 'xmin', 'xmax', 'ymean',
       'ydev', 'ymin', 'ymax', 'zmean', 'zdev', 'zmin', 'zmax','xrange', 'yrange', 'zrange']
#df2 = pd.DataFrame(df[['Class','user']])
#for name in columns:
#    minimum = df1[name].min()
#    maximum = df1[name].max()
#    df2[name] = (df[name]-minimum)/(maximum-minimum)
#
#df2.to_csv('data/normalize test features.csv', index=False)
#
df2 = pd.DataFrame(df[['Class','user']])
for name in columns:
    meanvalue = df1[name].mean()
    stdev = df1[name].std()
    df2[name] = (df[name]-meanvalue)/stdev

df2.to_csv('data/standardize test features.csv', index=False)


#
#
#
##