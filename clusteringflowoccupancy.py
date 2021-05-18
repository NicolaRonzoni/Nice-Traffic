#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:35:16 2021

@author: nicolaronzoni
"""

import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random

import matplotlib.pyplot as plt
fig = plt.gcf()

x= np.arange(0,24,0.1)

#########TRAIN DATA##########
#import  train data
df= pd.read_excel(r"C094 6min 2019.xlsx") 

df

plt.plot(df["occupancy"],df["flow"], ',' )
plt.xlabel(xlabel='occupancy rate')
plt.ylabel(ylabel='flow')
plt.title(label='fundamental diagram')

flow = pd.Series(data=df["flow"].values, index=df["index"])

flow.plot(title='2020 series C01 detector')

flow.plot(kind='hist')

occupancy = pd.Series(data=df["occupancy"].values, index=df["index"])

occupancy.plot(title='2020 series C601 detector')

occupancy.plot(kind='hist')


###### descriptive analysis 
#from 1/10 to 31/12
df1=df[29039:51119]
len(df1)
#from 1/10 to 31/10
df2=df[29039:36479]
len(df2)
#from 1/11 to 30/11
df3=df[36479:43679]
len(df3)
#from 1/10 to 30/11
df4=df[29039:43679]
len(df4)
#from 4/11 to 21/12
df5=df[37440:48960]
len(df5)
#from 9/9 to 19/10
df6=df[24000:33840]
len(df6)
#from 2/9 to 21/12
df7=df[22320:48960]
len(df7)
#from 2/9 to 19/10
df8=df[22320:33840]
len(df8)

#from 2/9 to 19/10 & from 4/11 to 21/12
df9= [df8,df5]
result= pd.concat(df9)
len(result)


#2020 from 1/9 to 18/10 
df11=df[49920:61440]
len(df11)

#treatment of flow variables 
flow=df7.loc[:,'flow']
flow
#normalization/standardization of train data 
flow=np.array(flow)
flow= flow.reshape((len(flow), 1))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler=StandardScaler()
#fit train data 
scaler_flow_train = scaler.fit(flow)
print('Min: %f, Max: %f' % (scaler_flow_train.data_min_, scaler_flow_train.data_max_))
#scale train data 
normalized_flow=scaler_flow_train.transform(flow)
normalized_flow
#from array to list 
normalized_flow=normalized_flow.tolist()
len(normalized_flow)
from toolz.itertoolz import sliding_window, partition
#for every day of the train set store the flow observations 
day_flow=list(partition(240,normalized_flow))
day_flow
len(day_flow)
#from list to multidimensional array 
day_flow=np.asarray(day_flow)
day_flow
from tslearn.utils import to_time_series, to_time_series_dataset
#create univariate series for normalized flow_observation 
first_time_series = to_time_series(day_flow)
print(first_time_series.shape)

#treatment of occupacy variable 
occupacy =df7.loc[:,'occupancy']
occupacy
#normalization/standardization of train data 
occupacy=np.array(occupacy)
occupacy= occupacy.reshape((len(occupacy), 1))
#fit train data
scaler_occupacy_train = scaler.fit(occupacy)
print('Min: %f, Max: %f' % (scaler_occupacy_train.data_min_, scaler_occupacy_train.data_max_))
#scale train data 
normalized_occupacy = scaler_occupacy_train.transform(occupacy)
normalized_occupacy
#from array to list 
normalized_occupacy=normalized_occupacy.tolist()
len(normalized_occupacy)
#for every day of the train set store the speed observations 
day_occupacy=list(partition(240,normalized_occupacy))
day_occupacy
len(day_occupacy)
#from list to multidimensionalarray
day_occupacy=np.asarray(day_occupacy)
day_occupacy
#create univariate series for normalized speed observation 
second_time_series = to_time_series(day_occupacy)
print(second_time_series.shape)

#create the multivariate time series TRAIN  
multivariate=np.dstack((first_time_series,second_time_series))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#######TEST DATA ########
#import  test data
df_test= pd.read_excel(r"C009 6min 2019.xlsx") 
df_test

df10=df_test[22320:48960]
len(df10)
#treatment of flow variables 
flow_test=df10.loc[:,'flow']
#normalization/standardization of train data 
flow_test=np.array(flow_test)
flow_test= flow_test.reshape((len(flow_test), 1))

#fit test data 
scaler_flow_test = scaler.fit(flow_test)
print('Min: %f, Max: %f' % (scaler_flow_test.data_min_, scaler_flow_test.data_max_))
#scale test data 
normalized_flow_test=scaler_flow_test.transform(flow_test)
normalized_flow_test
#from array to list 
normalized_flow_test=normalized_flow_test.tolist()
len(normalized_flow_test)

#for every day of the test set store the flow observations 
day_flow_test=list(partition(240,normalized_flow_test))
day_flow_test
len(day_flow_test)
#from list to multidimensional array 
day_flow_test=np.asarray(day_flow_test)
day_flow_test
#create univariate series for normalized flow_observations of the test set 
first_time_series_test = to_time_series(day_flow_test)
print(first_time_series_test.shape)

#treatment of density variable 
density_test =df10.loc[:,'occupancy']
#normalization/standardization of test data 
density_test=np.array(density_test)
density_test= density_test.reshape((len(density_test), 1))
#fit test data
scaler_density_test = scaler.fit(density_test)
print('Min: %f, Max: %f' % (scaler_density_test.data_min_, scaler_density_test.data_max_))
#scale train data 
normalized_density_test = scaler_density_test.transform(density_test)
normalized_density_test
#from array to list 
normalized_density_test=normalized_density_test.tolist()
len(normalized_density_test)
#for every day of the test set store the speed observations 
day_density_test=list(partition(240,normalized_density_test))
day_density_test
len(day_density_test)
#from list to multidimensionalarray
day_density_test=np.asarray(day_density_test)
day_density_test
#create univariate series for normalized speed observations of the test set  
second_time_series_test = to_time_series(day_density_test)
print(second_time_series_test.shape)

#create the multivariate time series TEST 
multivariate_test=np.dstack((first_time_series_test,second_time_series_test))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)


#clustering 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
############################### soft dynamic time warping ##############################
###### grid search for times
### list of hyperparameters considered 
silhouette=[]
gamma=[0.1,0.3,0.5,0.8,1]
for i in range(0,5):
     k=TimeSeriesKMeans(n_clusters=, metric="softdtw",metric_params={"gamma": gamma[i]}, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)
     prediction=k.fit_predict(multivariate_time_series_train,y=None)
     s=silhouette_score(multivariate_time_series_train, prediction, metric="softdtw",metric_params={"gamma": gamma[i]})
     silhouette.append(s)
    
silhouette   
from tslearn.metrics import gamma_soft_dtw
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the algorithm on train data based on the previous grid search
#tune the hyperparameters possible metrics: euclidean, dtw, softdtw
km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw",metric_params={"gamma": 22.206778280673635}, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)
km_dba.cluster_centers_.shape
#prediction on train data 
prediction_train= km_dba.fit_predict(multivariate_time_series_train,y=None)
len(prediction_train)
#prediction on test data 
prediction_test= km_dba.predict(multivariate_time_series_test)
len(prediction_test)
prediction_test

#accuracy of the clustering on the train data 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma": 22.206778280673635})
#accuracy of the clustering on the test data
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma": 22.206778280673635})
############################################ k=2 #########################################
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]

random.shuffle(cluster2)



sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)


#plot centroids k=2 

centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel(' hours of the day')
plt.ylabel('flow')
plt.title(' k=0')
plt.legend()
plt.subplot(2,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupacy')
plt.xlabel(' hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(2,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title(' k=1')
plt.legend()
plt.subplot(2,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#ff3399', label = 'occupacy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.suptitle("centroids path softDTW", fontsize=30)
plt.show()

########################################### k=3 ################################
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:10]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)


# plot the centroids k=3 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title(' k=0')
plt.legend()
plt.subplot(3,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#6666ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#6666ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,5)
for i in range(0,10):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(3,2,6)
for i in range(0,10):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_3),'#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=2')
plt.legend()
plt.suptitle("centroids path softDTW", fontsize=30)
plt.show()

###################################### k=4 ######################@@@@@###########
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]
cluster1.shape
random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:15]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_occupacy_train.inverse_transform(sample4density)



# plot the centroids K=4 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids[3][:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids[3][:,1]
density_4= density_4.reshape((len(density_4), 1))
plt.figure(figsize=(20,30))
plt.subplot(3,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#3399ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#3399ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()

plt.subplot(3,2,5)
for i in range(0,15):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(3,2,6)
for i in range(0,15):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_4),'#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=3')
plt.legend()
plt.suptitle("centroids path softDTW", fontsize=30)
plt.show()


###################################### k=5 ######################@@@@@###########
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow.shape

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:15]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow.shape

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:15]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_occupacy_train.inverse_transform(sample4density)

#select randomly time series from fifth cluster 

cluster5=multivariate_time_series_train[prediction_train==4]

random.shuffle(cluster5)

sample5=cluster5[0:15]

sample5.shape

sample5flow=sample5[:,:,0]
sample5flow=scaler_flow_train.inverse_transform(sample5flow)
sample5flow

sample5density=sample5[:,:,1]
sample5density=scaler_occupacy_train.inverse_transform(sample5density)



# plot the centroids K=5 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids[3][:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
flow_5=centroids[4][:,0]
flow_5= flow_5.reshape((len(flow_5), 1))

density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids[3][:,1]
density_4= density_4.reshape((len(density_4), 1))
density_5=centroids[4][:,1]
density_5= density_5.reshape((len(density_5), 1))
plt.figure(figsize=(20,30))
plt.subplot(5,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#00b8e6', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#00b8e6', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,5)
for i in range(0,15):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'#4d4dff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,6)
for i in range(0,15):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_3),'#4d4dff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,7)
for i in range(0,15):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'#cc00ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(5,2,8)
for i in range(0,15):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_4),'#cc00ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=3')
plt.subplot(5,2,9)
for i in range(0,15):
    plt.plot(x,sample5flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_5),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=4')
plt.legend()
plt.subplot(5,2,10)
for i in range(0,15):
    plt.plot(x,sample5density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_5), '#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=4')
plt.legend()
plt.suptitle("centroids path softDTW", fontsize=30)
plt.show()



######################## GLOBAL ALIGNMENT KERNEL ################################

############################################ k=2 #########################################


#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

#compute centroids of the first group
from tslearn.barycenters import softdtw_barycenter
centroids1=softdtw_barycenter(cluster1, max_iter=5,gamma= 22.206778280673635)


random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]
centroids2=softdtw_barycenter(cluster2, max_iter=5,gamma= 22.206778280673635)
random.shuffle(cluster2)



sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)


#plot centroids k=2 
flow_1=centroids1[:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids2[:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
density_1=centroids1[:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids2[:,1]
density_2= density_2.reshape((len(density_2), 1))



plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel(' hours of the day')
plt.ylabel('flow')
plt.title(' k=0')
plt.legend()
plt.subplot(2,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupacy')
plt.xlabel(' hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(2,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title(' k=1')
plt.legend()
plt.subplot(2,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#ff3399', label = 'occupacy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.suptitle("centroids path after Kernel K-Means", fontsize=30)
plt.show()

########################################### k=3 ################################

cluster1=multivariate_time_series_train[prediction_train==0]
centroids1=softdtw_barycenter(cluster1, max_iter=5,gamma= 22.206778280673635)

random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]
centroids2=softdtw_barycenter(cluster2, max_iter=5,gamma=22.206778280673635)

random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]
centroids3=softdtw_barycenter(cluster3, max_iter=5,gamma= 22.206778280673635)
random.shuffle(cluster3)

sample3=cluster3[0:10]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)


# plot the centroids k=3 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids1[:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids2[:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids3[:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
density_1=centroids1[:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids2[:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids3[:,1]
density_3= density_3.reshape((len(density_3), 1))
plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title(' k=0')
plt.legend()
plt.subplot(3,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#6666ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#6666ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,5)
for i in range(0,10):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(3,2,6)
for i in range(0,10):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_3),'#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=2')
plt.legend()
plt.suptitle("centroids path after Kernel K-Means", fontsize=30)
plt.show()

###################################### k=4 ######################@@@@@###########

cluster1=multivariate_time_series_train[prediction_train==0]
cluster1.shape
centroids1=softdtw_barycenter(cluster1, max_iter=5,gamma= 23.599162083999055)
random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]
centroids2=softdtw_barycenter(cluster2, max_iter=5,gamma= 23.599162083999055)

random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]
centroids3=softdtw_barycenter(cluster3, max_iter=5,gamma= 23.599162083999055)
random.shuffle(cluster3)

sample3=cluster3[0:15]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]
centroids4=softdtw_barycenter(cluster4, max_iter=5,gamma=23.599162083999055)
random.shuffle(cluster4)

sample4=cluster4[0:15]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_occupacy_train.inverse_transform(sample4density)



# plot the centroids K=4 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids1[:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids2[:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids3[:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids4[:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
density_1=centroids1[:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids2[:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids3[:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids4[:,1]
density_4= density_4.reshape((len(density_4), 1))
plt.figure(figsize=(20,30))
plt.subplot(4,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#3399ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#3399ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,5)
for i in range(0,15):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'#cc33ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,6)
for i in range(0,15):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_3),'#cc33ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,7)
for i in range(0,15):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(4,2,8)
for i in range(0,15):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_4),'#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=3')
plt.legend()
plt.suptitle("centroids path after Kernel K-Means", fontsize=30)
plt.show()


###################################### k=5 ######################@@@@@###########

cluster1=multivariate_time_series_train[prediction_train==0]
centroids1=softdtw_barycenter(cluster1, max_iter=5,gamma= 22.206778280673635)
random.shuffle(cluster1)

sample1=cluster1[0:15]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_occupacy_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]
centroids2=softdtw_barycenter(cluster2, max_iter=5,gamma=22.206778280673635)

random.shuffle(cluster2)

sample2=cluster2[0:9]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow.shape

sample2density=sample2[:,:,1]
sample2density=scaler_occupacy_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]
centroids3=softdtw_barycenter(cluster3, max_iter=5,gamma= 22.206778280673635)
random.shuffle(cluster3)

sample3=cluster3

sample3.shape[0:15]

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow.shape

sample3density=sample3[:,:,1]
sample3density=scaler_occupacy_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]
centroids4=softdtw_barycenter(cluster4, max_iter=5,gamma=22.206778280673635)
random.shuffle(cluster4)

sample4=cluster4[0:15]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_occupacy_train.inverse_transform(sample4density)

#select randomly time series from fifth cluster 

cluster5=multivariate_time_series_train[prediction_train==4]
centroids5=softdtw_barycenter(cluster5, max_iter=5,gamma=22.206778280673635)
random.shuffle(cluster5)

sample5=cluster5[0:15]

sample5.shape

sample5flow=sample5[:,:,0]
sample5flow=scaler_flow_train.inverse_transform(sample5flow)
sample5flow

sample5density=sample5[:,:,1]
sample5density=scaler_occupacy_train.inverse_transform(sample5density)



# plot the centroids K=5 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids1[:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids2[:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids3[:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids4[:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
flow_5=centroids5[:,0]
flow_5= flow_5.reshape((len(flow_5), 1))

density_1=centroids1[:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids2[:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids3[:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids4[:,1]
density_4= density_4.reshape((len(density_4), 1))
density_5=centroids5[:,1]
density_5= density_5.reshape((len(density_5), 1))
plt.figure(figsize=(20,30))
plt.subplot(5,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'c-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_1),'c-', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,3)
for i in range(0,9):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'#00b8e6', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,4)
for i in range(0,9):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_2),'#00b8e6', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,5)
for i in range(0,15):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'#4d4dff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,6)
for i in range(0,15):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_3),'#4d4dff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,7)
for i in range(0,15):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'#cc00ff', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(5,2,8)
for i in range(0,15):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_4),'#cc00ff', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=3')
plt.subplot(5,2,9)
for i in range(0,15):
    plt.plot(x,sample5flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_5),'#ff3399', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=4')
plt.legend()
plt.subplot(5,2,10)
for i in range(0,15):
    plt.plot(x,sample5density[i],'k-', alpha=.2)
plt.plot(x,scaler_occupacy_train.inverse_transform(density_5), '#ff3399', label = 'occupancy')
plt.xlabel('hours of the day')
plt.ylabel('occupancy')
plt.title('k=4')
plt.legend()
plt.suptitle("centroids path after Kernel K-Means", fontsize=30)
plt.show()

##################################### global alignment kernel ###########################
gak= KernelKMeans(n_clusters=5, kernel="gak",kernel_params={"sigma":"auto"},max_iter=5, random_state=0).fit(multivariate_time_series_train) 

#train
prediction_train=gak.fit_predict(multivariate_time_series_train)
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":22.206778280673635})

#test 
prediction_test=gak.fit_predict(multivariate_time_series_test)
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma":22.206778280673635})

#similarity between centroids of the clusters 
from tslearn.metrics import soft_dtw, cdist_soft_dtw
similarity=[]
matrix=cdist_soft_dtw(centroids, gamma=1.)
matrix
sim=matrix.max()
similarity.append(sim)

similarity=np.array(similarity)
diss=list(-similarity)

cluster=np.arange(2,8)
plt.title('soft-DTW similarity measure S60')
plt.plot(cluster,diss)
plt.xlabel('n° of cluster')
plt.ylabel('similarity between closest clusters')
plt.show()

#visualization 
import calplot

#all days of 2013 
first_mid= pd.date_range('9/2/2019', periods=48, freq='D')
second_mid= pd.date_range('11/4/2019', periods=48, freq='D')
#assign at every day the cluster 
events_train_first = pd.Series(prediction_train[0:48],index=first_mid)
events_train_second = pd.Series(prediction_train[48:96],index=second_mid)
series=[events_train_first,events_train_second]

events_train=pd.concat(series)

index_all= pd.date_range('2/6/2020', periods=277, freq='D')

#plot the result train 
events_train = pd.Series(prediction_train,index=index_all)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days train set C601 softDTW', linewidth=2.3)  
new=[]
for i in range(0,111):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)


#plot the result test  
events_test = pd.Series(prediction_test,index=index_all)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days test set C009 softDTW', linewidth=2.3)  
new=[]
for i in range(0,111):
    if prediction_test[i] == 0:
        y=0.05
    elif prediction_test[i] !=0: 
        y=prediction_test[i]
    new.append(y)
    






