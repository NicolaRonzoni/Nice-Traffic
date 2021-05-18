#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:43:11 2021

@author: nronzoni
"""

import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random

##### strategy for normalization 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler= StandardScaler()

##### sliding window for create daily time series
from toolz.itertoolz import sliding_window, partition
##### 
from tslearn.utils import to_time_series, to_time_series_dataset
#########TRAIN DATA##########

#import  train data

df= pd.read_excel(r"firstmidtermflow.xlsx") 

df

#create a univariate time series for every detector
#### C601 ####
C601=df.loc[:,'C601']
C601
#normalization/standardization of train data 
C601=np.array(C601)
C601= C601.reshape((len(C601), 1))
#fit train data 
scaler_C601_train = scaler.fit(C601)
#print('Min: %f, Max: %f' % (scaler_C601_train.data_min_, scaler_C601_train.data_max_))
#scale train data 
normalized_C601=scaler_C601_train.transform(C601)
normalized_C601
#from array to list 
normalized_C601=normalized_C601.tolist()
len(normalized_C601)
#for every day of the train set store the flow observations 
day_C601=list(partition(240,normalized_C601))
day_C601
len(day_C601)
#from list to multidimensional array 
day_C601=np.asarray(day_C601)
day_C601
#create univariate series for normalized flow_observation 
C601_time_series = to_time_series(day_C601)

print(C601_time_series.shape)

#### C602 ####
C602=df.loc[:,'C602']
C602
#normalization/standardization of train data 
C602=np.array(C602)
C602= C602.reshape((len(C602), 1))
#fit train data 
scaler_C602_train = scaler.fit(C602)
#print('Min: %f, Max: %f' % (scaler_C602_train.data_min_, scaler_C602_train.data_max_))
#scale train data 
normalized_C602=scaler_C602_train.transform(C602)
normalized_C602
#from array to list 
normalized_C602=normalized_C602.tolist()
len(normalized_C602)
#for every day of the train set store the flow observations 
day_C602=list(partition(240,normalized_C602))
day_C602
len(day_C602)
#from list to multidimensional array 
day_C602=np.asarray(day_C602)
day_C602
#create univariate series for normalized flow_observation 
C602_time_series = to_time_series(day_C602)

print(C602_time_series.shape)

#### C614 ####
C614=df.loc[:,'C614']
C614
#normalization/standardization of train data 
C614=np.array(C614)
C614= C614.reshape((len(C614), 1))
#fit train data 
scaler_C614_train = scaler.fit(C614)
#print('Min: %f, Max: %f' % (scaler_C614_train.data_min_, scaler_C614_train.data_max_))
#scale train data 
normalized_C614=scaler_C614_train.transform(C614)
normalized_C614
#from array to list 
normalized_C614=normalized_C614.tolist()
len(normalized_C614)
#for every day of the train set store the flow observations 
day_C614=list(partition(240,normalized_C614))
day_C614
len(day_C614)
#from list to multidimensional array 
day_C614=np.asarray(day_C614)
day_C614
#create univariate series for normalized flow_observation 
C614_time_series = to_time_series(day_C614)

print(C614_time_series.shape)

#### C009 ####
C009=df.loc[:,'C009']
C009
#normalization/standardization of train data 
C009=np.array(C009)
C009= C009.reshape((len(C009), 1))
#fit train data 
scaler_C009_train = scaler.fit(C009)
#print('Min: %f, Max: %f' % (scaler_C009_train.data_min_, scaler_C009_train.data_max_))
#scale train data 
normalized_C009=scaler_C009_train.transform(C009)
normalized_C009
#from array to list 
normalized_C009=normalized_C009.tolist()
len(normalized_C009)
#for every day of the train set store the flow observations 
day_C009=list(partition(240,normalized_C009))
day_C009
len(day_C009)
#from list to multidimensional array 
day_C009=np.asarray(day_C009)
day_C009
#create univariate series for normalized flow_observation 
C009_time_series = to_time_series(day_C009)

print(C009_time_series.shape)

#### C094 ####
C094=df.loc[:,'C094']
C094
#normalization/standardization of train data 
C094=np.array(C094)
C094= C094.reshape((len(C094), 1))
#fit train data 
scaler_C094_train = scaler.fit(C094)
#print('Min: %f, Max: %f' % (scaler_C094_train.data_min_, scaler_C094_train.data_max_))
#scale train data 
normalized_C094=scaler_C094_train.transform(C094)
normalized_C094
#from array to list 
normalized_C094=normalized_C094.tolist()
len(normalized_C094)
#for every day of the train set store the flow observations 
day_C094=list(partition(240,normalized_C094))
day_C094
len(day_C094)
#from list to multidimensional array 
day_C094=np.asarray(day_C094)
day_C094
#create univariate series for normalized flow_observation 
C094_time_series = to_time_series(day_C094)

print(C094_time_series.shape)

#### C599 ####
C599=df.loc[:,'C599']
C599
#normalization/standardization of train data 
C599=np.array(C599)
C599= C599.reshape((len(C599), 1))
#fit train data 
scaler_C599_train = scaler.fit(C599)
#print('Min: %f, Max: %f' % (scaler_C599_train.data_min_, scaler_C599_train.data_max_))
#scale train data 
normalized_C599=scaler_C599_train.transform(C599)
normalized_C599
#from array to list 
normalized_C599=normalized_C599.tolist()
len(normalized_C599)
#for every day of the train set store the flow observations 
day_C599=list(partition(240,normalized_C599))
day_C599
len(day_C599)
#from list to multidimensional array 
day_C599=np.asarray(day_C599)
day_C599
#create univariate series for normalized flow_observation 
C599_time_series = to_time_series(day_C599)

print(C599_time_series.shape)

########### create the multivariate time series 
multivariate=np.dstack((C601_time_series,C602_time_series,C009_time_series,C094_time_series,C614_time_series,C599_time_series))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#clustering 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score

gak=KernelKMeans(n_clusters=6, kernel="gak",kernel_params={"sigma":'auto'},max_iter=5,random_state=0).fit(multivariate_time_series_train)

prediction=gak.fit_predict(multivariate_time_series_train,y=None)

from tslearn.metrics import gamma_soft_dtw
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

km_dba = TimeSeriesKMeans(n_clusters=5, metric="softdtw",metric_params={"gamma":140.8349332772751}, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

prediction= km_dba.fit_predict(multivariate_time_series_train,y=None)
#visualization 

import calplot

#all days of 2013 
all_days= pd.date_range('9/1/2020', periods=48, freq='D')
#assign at every day the cluster 
events_train = pd.Series(prediction,index=all_days)
#plot the result 
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days softDTW flow', linewidth=2.3)  


new=[]
for i in range(0,48):
    if prediction[i] == 0:
        y=0.05
    elif prediction[i] !=0: 
        y=prediction[i]
    new.append(y)

