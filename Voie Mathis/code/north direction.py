#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:43:47 2021

@author: nronzoni
"""
import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random
from toolz.itertoolz import sliding_window, partition
from tslearn.utils import to_time_series, to_time_series_dataset
##### strategy for normalization 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.metrics import soft_dtw, gamma_soft_dtw,dtw
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from tslearn.generators import random_walks
from sklearn.pipeline import Pipeline

#IMPORT DATA North Direction 

cimiez_nord=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=0) 

philippe_nord=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=1)

magnan=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=2)

fabron_sortie=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=3)

bosquets_entree=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=4)

grinda=pd.read_excel(r"Voie Mathis North direction.xlsx",sheet_name=5)
     
series=pd.Series(data=grinda["speed"].values,index=grinda.index)

series.plot()


#cimiez nord
cimiez_nord_flow=data_split_flow_north(cimiez_nord)
## create daily time series train 
series_train_cimiez_nord_flow=daily_series(cimiez_nord_flow[0],170)
series_train_cimiez_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_nord_flow[1].data_min_, series_train_cimiez_nord_flow[1].data_max_))
#philippe nord
philippe_nord_flow=data_split_flow_north(philippe_nord)
## create daily time series train 
series_train_philippe_nord_flow=daily_series(philippe_nord_flow[0],170)
series_train_philippe_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_nord_flow[1].data_min_, series_train_philippe_nord_flow[1].data_max_))
#magnan
magnan_flow=data_split_flow_north(magnan)
## create daily time series train 
series_train_magnan_flow=daily_series(magnan_flow[0],170)
series_train_magnan_flow[0].shape
print('Min: %f, Max: %f' % (series_train_magnan_flow[1].data_min_, series_train_magnan_flow[1].data_max_))
#fabron sortie
fabron_sortie_flow=data_split_flow_north(fabron_sortie)
## create daily time series train 
series_train_fabron_sortie_flow=daily_series(fabron_sortie_flow[0],170)
series_train_fabron_sortie_flow[0].shape
print('Min: %f, Max: %f' % (series_train_fabron_sortie_flow[1].data_min_, series_train_fabron_sortie_flow[1].data_max_))
#bosquets entree
bosquets_entree_flow=data_split_flow_north(bosquets_entree)
## create daily time series train 
series_train_bosquets_entree_flow=daily_series(bosquets_entree_flow[0],170)
series_train_bosquets_entree_flow[0].shape
print('Min: %f, Max: %f' % (series_train_bosquets_entree_flow[1].data_min_, series_train_bosquets_entree_flow[1].data_max_))
#grinda
#grinda
grinda_flow=data_split_flow_north(grinda)
## create daily time series train 
series_train_grinda_flow=daily_series(grinda_flow[0],170)
series_train_grinda_flow[0].shape
print('Min: %f, Max: %f' % (series_train_grinda_flow[1].data_min_, series_train_grinda_flow[1].data_max_))


#multivariate time series train
multivariate=np.dstack((series_train_cimiez_nord_flow[0],series_train_philippe_nord_flow[0],series_train_magnan_flow[0],series_train_grinda_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)


#cimiez nord 
series_test_cimiez_nord_flow=daily_series(cimiez_nord_flow[1],170)
series_test_cimiez_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_test_cimiez_nord_flow[1].data_min_, series_test_cimiez_nord_flow[1].data_max_))
#philippe nord
series_test_philippe_nord_flow=daily_series(philippe_nord_flow[1],170)
series_test_philippe_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_test_philippe_nord_flow[1].data_min_, series_test_philippe_nord_flow[1].data_max_))
#magnan
series_test_magnan_flow=daily_series(magnan_flow[1],170)
series_test_magnan_flow[0].shape
print('Min: %f, Max: %f' % (series_test_magnan_flow[1].data_min_, series_test_magnan_flow[1].data_max_))
#fabron sortie
series_test_fabron_sortie_flow=daily_series(fabron_sortie_flow[1],170)
series_test_fabron_sortie_flow[0].shape
print('Min: %f, Max: %f' % (series_test_fabron_sortie_flow[1].data_min_, series_test_fabron_sortie_flow[1].data_max_))
#bosquets entree
series_test_bosquets_entree_flow=daily_series(bosquets_entree_flow[1],170)
series_test_bosquets_entree_flow[0].shape
print('Min: %f, Max: %f' % (series_test_bosquets_entree_flow[1].data_min_, series_test_bosquets_entree_flow[1].data_max_))
#grinda
series_test_grinda_flow=daily_series(grinda_flow[1],170)
series_test_grinda_flow[0].shape
print('Min: %f, Max: %f' % (series_test_grinda_flow[1].data_min_, series_test_grinda_flow[1].data_max_))


#multivariate time series test
multivariate_test=np.dstack((series_test_cimiez_nord_flow[0],series_test_philippe_nord_flow[0],series_test_magnan_flow[0],series_test_grinda_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#CLUSTERING

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw

score_g, df = optimalK(multivariate_time_series_train, nrefs=5, maxClusters=7)

plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. number of cluster, test set');

np.argwhere(np.isnan(multivariate_time_series_train))
#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train,y=None)

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test)

#silhouette
#train 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw",metric_params={"gamma":})
#test 
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw",metric_params={"gamma":})


#visualization
import calplot

# north direction 

#train
# from 1/1 to 13/1 
first_mid=pd.date_range('1/1/2019', periods=13, freq='D')
#from 15/1 to 16/1
second_mid=pd.date_range('1/15/2019', periods=2, freq='D')
#from 18/1 to  26/1
third_mid=pd.date_range('1/18/2019', periods=9, freq='D')
#from 28/1 to  2/2
third_mid_bis=pd.date_range('1/28/2019', periods=6, freq='D')
#from 8/3 to 8/4
fourth_mid=pd.date_range('3/8/2019', periods=32, freq='D')
# from 11/4 to 23/4
fifth_mid=pd.date_range('4/11/2019', periods=13, freq='D')
# from 25/4 to 21/5
sixth_mid=pd.date_range('4/25/2019', periods=27, freq='D')
#from 23/5 to 1/7
septh_mid=pd.date_range('5/23/2019', periods=40, freq='D')
#from 4/7 to  24/8
eight_mid=pd.date_range('7/4/2019', periods=52, freq='D')
# from 26/8 to 26/9
ninth_mid=pd.date_range('8/26/2019', periods=32, freq='D')
# from 28/9 to 3/10
tenth_mid=pd.date_range('9/28/2019', periods=6, freq='D')
# from 5/10 to 16/10
eleventh_mid=pd.date_range('10/5/2019', periods=12, freq='D')
#from 18/10 to 23/10
twelveth_mid=pd.date_range('10/18/2019', periods=6, freq='D')
# from 25/10 to 12/11
thirdteen_mid=pd.date_range('10/25/2019', periods=19, freq='D')
# from 14/11 to 24/11
fourthteen_mid=pd.date_range('11/14/2019', periods=11, freq='D')
#from 27/11 to 8/12
sixtheen_mid=pd.date_range('11/27/2019', periods=12, freq='D')
#from 10/12 to 17/12
seventeen_mid=pd.date_range('12/10/2019', periods=8, freq='D')
#from 19/12 to 19/12 
eighteen_mid=pd.date_range('12/19/2019', periods=1, freq='D')
#from 21/12 to 31/12 
eighteen_mid_bis=pd.date_range('12/21/2019', periods=11, freq='D')


first_mid=pd.Series(data=first_mid)
second_mid=pd.Series(data=second_mid)
third_mid=pd.Series(data=third_mid)
third_mid_bis=pd.Series(data=third_mid_bis)
fourth_mid=pd.Series(data=fourth_mid)
fifth_mid=pd.Series(data=fifth_mid)
sixth_mid=pd.Series(data=sixth_mid)
septh_mid=pd.Series(data=septh_mid)
eight_mid=pd.Series(data=eight_mid)
ninth_mid=pd.Series(data=ninth_mid)
tenth_mid=pd.Series(data=tenth_mid)
eleventh_mid=pd.Series(data=eleventh_mid)
twelveth_mid=pd.Series(data=twelveth_mid)
thirdteen_mid=pd.Series(data=thirdteen_mid)
fourthteen_mid=pd.Series(data=fourthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
eighteen_mid_bis=pd.Series(data=eighteen_mid_bis)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid,eighteen_mid_bis],ignore_index=True)
len(index_train)
index_train[0]

# north direction  bis 

#train
# from 1/1 to 13/1 
first_mid=pd.date_range('1/1/2019', periods=13, freq='D')
#from 15/1 to 16/1
second_mid=pd.date_range('1/15/2019', periods=2, freq='D')
#from 18/1 to  26/1
third_mid=pd.date_range('1/18/2019', periods=9, freq='D')
#from 28/1 to  13/2
third_mid_bis=pd.date_range('1/28/2019', periods=17, freq='D')
#from 16/2 to  18/2
third_mid_tris=pd.date_range('2/16/2019', periods=3, freq='D')
#from 20/2 to 8/4
fourth_mid=pd.date_range('2/20/2019', periods=48, freq='D')
# from 11/4 to 23/4
fifth_mid=pd.date_range('4/11/2019', periods=13, freq='D')
# from 25/4 to 21/5
sixth_mid=pd.date_range('4/25/2019', periods=27, freq='D')
#from 23/5 to 1/7
septh_mid=pd.date_range('5/23/2019', periods=40, freq='D')
#from 4/7 to  24/8
eight_mid=pd.date_range('7/4/2019', periods=52, freq='D')
# from 26/8 to 26/9
ninth_mid=pd.date_range('8/26/2019', periods=32, freq='D')
# from 28/9 to 3/10
tenth_mid=pd.date_range('9/28/2019', periods=6, freq='D')
# from 5/10 to 16/10
eleventh_mid=pd.date_range('10/5/2019', periods=12, freq='D')
#from 18/10 to 23/10
twelveth_mid=pd.date_range('10/18/2019', periods=6, freq='D')
# from 25/10 to 12/11
thirdteen_mid=pd.date_range('10/25/2019', periods=19, freq='D')
# from 14/11 to 24/11
fourthteen_mid=pd.date_range('11/14/2019', periods=11, freq='D')
#from 27/11 to 8/12
sixtheen_mid=pd.date_range('11/27/2019', periods=12, freq='D')
#from 10/12 to 17/12
seventeen_mid=pd.date_range('12/10/2019', periods=8, freq='D')
#from 19/12 to 19/12 
eighteen_mid=pd.date_range('12/19/2019', periods=1, freq='D')
#from 21/12 to 31/12 
eighteen_mid_bis=pd.date_range('12/21/2019', periods=11, freq='D')


first_mid=pd.Series(data=first_mid)
second_mid=pd.Series(data=second_mid)
third_mid=pd.Series(data=third_mid)
third_mid_bis=pd.Series(data=third_mid_bis)
third_mid_tris=pd.Series(data=third_mid_tris)
fourth_mid=pd.Series(data=fourth_mid)
fifth_mid=pd.Series(data=fifth_mid)
sixth_mid=pd.Series(data=sixth_mid)
septh_mid=pd.Series(data=septh_mid)
eight_mid=pd.Series(data=eight_mid)
ninth_mid=pd.Series(data=ninth_mid)
tenth_mid=pd.Series(data=tenth_mid)
eleventh_mid=pd.Series(data=eleventh_mid)
twelveth_mid=pd.Series(data=twelveth_mid)
thirdteen_mid=pd.Series(data=thirdteen_mid)
fourthteen_mid=pd.Series(data=fourthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
eighteen_mid_bis=pd.Series(data=eighteen_mid_bis)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,third_mid_tris,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid,eighteen_mid_bis],ignore_index=True)
len(index_train)

#plot the result 
new=[]
for i in range(0,312):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
    
for i in range(0,312):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,312):
    if new[i] == 2:
        new[i] =0.05

for i in range(0,312):
    if new[i] ==4:
        new[i] =2

#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis North Direction 2019 (train):speed loops', linewidth=2.3,dropzero=True,vmin=0) 

for i in range(0,357):
    if new[i] == 0.05:
        new[i]=0
    prediction_train[i]=new[i]

np.set_printoptions(threshold=400)        
prediction_train      


#test
first_week=pd.date_range('1/20/2020', periods=7, freq='D')
second_week=pd.date_range('2/24/2020', periods=7, freq='D')
third_week=pd.date_range('3/23/2020', periods=7, freq='D')
fourth_week=pd.date_range('7/20/2020', periods=7, freq='D')


first_week=pd.Series(data=first_week)
second_week=pd.Series(data=second_week)
third_week=pd.Series(data=third_week)
fourth_week=pd.Series(data=fourth_week)


index_test=pd.concat([first_week,second_week,third_week,fourth_week],ignore_index=True)
new=[]
for i in range(0,28):
    if prediction_test[i] == 0:
        y=0.05
    elif prediction_test[i] !=0: 
        y=prediction_test[i]
    new.append(y)
  
for i in range(0,28):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,28):
    if new[i] == 2:
        new[i] =0.05

for i in range(0,28):
    if new[i] ==4:
        new[i] =2


events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis North Direction 2020 (test):speed loops', linewidth=2.3,dropzero=True,vmin=0) 


for i in range(0,35):
    if new[i] == 0.05:
        new[i]=0
    prediction_test[i]=new[i]

np.set_printoptions(threshold=400)        
prediction_test      


#######################################
#plot of the centroid 

#centroids 
centroids=km_dba.cluster_centers_

centroids.shape

##### first cluster #######
cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:20]

sample1.shape

cimiez_nord_sample1=sample1[:,:,0]
cimiez_nord_sample1=series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_sample1)
cimiez_nord_sample1.shape

philippe_nord_sample1=sample1[:,:,1]
philippe_nord_sample1=series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_sample1)
philippe_nord_sample1.shape

magnan_sample1=sample1[:,:,2]
magnan_sample1=series_train_magnan_flow[1].inverse_transform(magnan_sample1)
magnan_sample1.shape

grinda_sample1=sample1[:,:,3]
grinda_sample1=series_train_grinda_flow[1].inverse_transform(grinda_sample1)
grinda_sample1.shape


####second cluster #######

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape

cimiez_nord_sample2=sample2[:,:,0]
cimiez_nord_sample2=series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_sample2)
cimiez_nord_sample2.shape

philippe_nord_sample2=sample2[:,:,1]
philippe_nord_sample2=series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_sample2)
philippe_nord_sample2.shape

magnan_sample2=sample2[:,:,2]
magnan_sample2=series_train_magnan_flow[1].inverse_transform(magnan_sample2)
magnan_sample2.shape

grinda_sample2=sample2[:,:,3]
grinda_sample2=series_train_grinda_flow[1].inverse_transform(grinda_sample2)
grinda_sample2.shape


#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape

cimiez_nord_sample3=sample3[:,:,0]
cimiez_nord_sample3=series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_sample3)
cimiez_nord_sample3.shape

philippe_nord_sample3=sample3[:,:,1]
philippe_nord_sample3=series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_sample3)
philippe_nord_sample3.shape

magnan_sample3=sample3[:,:,2]
magnan_sample3=series_train_magnan_flow[1].inverse_transform(magnan_sample3)
magnan_sample3.shape

grinda_sample3=sample3[:,:,3]
grinda_sample3=series_train_grinda_flow[1].inverse_transform(grinda_sample3)
grinda_sample3.shape



#k=0#
cimiez_nord_1=centroids[0][:,0]
cimiez_nord_1=cimiez_nord_1.reshape((len(cimiez_nord_1), 1))

philippe_nord_1=centroids[0][:,1]
philippe_nord_1=philippe_nord_1.reshape((len(philippe_nord_1), 1))

magnan_1=centroids[0][:,2]
magnan_1=magnan_1.reshape((len(magnan_1), 1))

grinda_1=centroids[0][:,3]
grinda_1=grinda_1.reshape((len(grinda_1), 1))


#k=1#
cimiez_nord_2=centroids[1][:,0]
cimiez_nord_2=cimiez_nord_2.reshape((len(cimiez_nord_2), 1))

philippe_nord_2=centroids[1][:,1]
philippe_nord_2=philippe_nord_2.reshape((len(philippe_nord_2), 1))

magnan_2=centroids[1][:,2]
magnan_2=magnan_2.reshape((len(magnan_2), 1))

grinda_2=centroids[1][:,3]
grinda_2=grinda_2.reshape((len(grinda_2), 1))


#k=2#
cimiez_nord_3=centroids[2][:,0]
cimiez_nord_3=cimiez_nord_3.reshape((len(cimiez_nord_3), 1))

philippe_nord_3=centroids[2][:,1]
philippe_nord_3=philippe_nord_3.reshape((len(philippe_nord_3), 1))

magnan_3=centroids[2][:,2]
magnan_3=magnan_3.reshape((len(magnan_3), 1))

grinda_3=centroids[2][:,3]
grinda_3=grinda_3.reshape((len(grinda_3), 1))


x=np.arange(6,23,0.1)
len(x)


plt.figure(figsize=(35,35))
plt.subplot(3,4,1)
for i in range(0,20):
    plt.plot(x,cimiez_nord_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_1),'#33cc33', label = 'Cimiez Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,2)
for i in range(0,20):
    plt.plot(x,philippe_nord_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_1),'#33cc33', label = 'Philippe Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,3)
for i in range(0,20):
    plt.plot(x,magnan_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_magnan_flow[1].inverse_transform(magnan_1),'#33cc33', label = 'Magnan',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,4)
for i in range(0,20):
    plt.plot(x,grinda_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_grinda_flow[1].inverse_transform(grinda_1),'#33cc33', label = 'Grinda',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,5)
for i in range(0,20):
    plt.plot(x,cimiez_nord_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_2),'#0033cc', label = 'Cimiez Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,6)
for i in range(0,20):
    plt.plot(x,philippe_nord_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_2),'#0033cc', label = 'Philippe Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,7)
for i in range(0,20):
    plt.plot(x,magnan_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_magnan_flow[1].inverse_transform(magnan_2),'#0033cc', label = 'Magnan',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,8)
for i in range(0,20):
    plt.plot(x,grinda_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_grinda_flow[1].inverse_transform(grinda_2),'#0033cc', label = 'Grinda',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,9)
for i in range(0,20):
    plt.plot(x,cimiez_nord_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_nord_flow[1].inverse_transform(cimiez_nord_3),'#666699', label = 'Cimiez Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,10)
for i in range(0,20):
    plt.plot(x,philippe_nord_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_nord_flow[1].inverse_transform(philippe_nord_3),'#666699', label = 'Philippe Nord',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,11)
for i in range(0,20):
    plt.plot(x,magnan_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_magnan_flow[1].inverse_transform(magnan_3),'#666699', label = 'Magnan',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,4,12)
for i in range(0,20):
    plt.plot(x,grinda_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_grinda_flow[1].inverse_transform(grinda_3),'#666699', label = 'Grinda',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('dèbit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.figtext(0.5,0.36, "Working days", ha="center", va="top", fontsize=24, color="r")
plt.figtext(0.5,0.63, "Sundays", ha="center", va="top", fontsize=24, color="r")
plt.figtext(0.5,0.90, "Saturdays, August ", ha="center", va="top", fontsize=24, color="r")
plt.show()


##################################### prediction
##################### speed

multivariate_time_series_train.shape

first_day_cimiez_nord=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[26:27,:,:],5,110,0)
first_day_philippe_nord=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[26:27,:,:],5,110,1)
first_day_magnan=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[26:27,:,:],5,110,2)
first_day_grinda=walk_forward_validation_bis(multivariate_time_series_train,multivariate_time_series_test[26:27,:,:],5,110,3)


Y_pred_cimiez_nord=series_test_cimiez_nord_flow[1].inverse_transform(first_day_cimiez_nord[0].reshape(-1,1))
Y_test_cimiez_nord=series_test_cimiez_nord_flow[1].inverse_transform(first_day_cimiez_nord[1].reshape(-1,1))
error_cimiez_nord=math.sqrt(mean_squared_error(Y_test_cimiez_nord.reshape(-1,1),Y_pred_cimiez_nord.reshape(-1,1)))
Y_pred_philippe_nord=series_test_philippe_nord_flow[1].inverse_transform(first_day_philippe_nord[0].reshape(-1,1))
Y_test_philippe_nord=series_test_philippe_nord_flow[1].inverse_transform(first_day_philippe_nord[1].reshape(-1,1))
error_philippe_nord=math.sqrt(mean_squared_error(Y_test_philippe_nord.reshape(-1,1),Y_pred_philippe_nord.reshape(-1,1)))
Y_pred_magnan=series_test_magnan_flow[1].inverse_transform(first_day_magnan[0].reshape(-1,1))
Y_test_magnan=series_test_magnan_flow[1].inverse_transform(first_day_magnan[1].reshape(-1,1))
error_magnan=math.sqrt(mean_squared_error(Y_test_magnan.reshape(-1,1),Y_pred_magnan.reshape(-1,1)))
Y_pred_grinda=series_test_grinda_flow[1].inverse_transform(first_day_grinda[0].reshape(-1,1))
Y_test_grinda=series_test_grinda_flow[1].inverse_transform(first_day_grinda[1].reshape(-1,1))
error_grinda=math.sqrt(mean_squared_error(Y_test_grinda.reshape(-1,1),Y_pred_grinda.reshape(-1,1)))


error=mean([error_cimiez_nord,error_philippe_nord,error_magnan,error_grinda])


columns = ['Cimiez Nord speed (km/h)','Cimiez Nord speed (km/h) ground truth','Philippe Nord speed (km/h)','Philippe Nord speed (km/h) ground truth','Magnan speed (km/h)','Magnan speed (km/h) ground truth','Grinda speed (km/h)','Grinda speed (km/h) ground truth']
index=pd.date_range("17:00", periods=20, freq="6min")
df_7= pd.DataFrame(index=index.time, columns=columns)
df_7
df_7['Cimiez Nord speed (km/h)']=Y_pred_cimiez_nord.reshape(-1,1)
df_7['Cimiez Nord speed (km/h) ground truth']=Y_test_cimiez_nord.reshape(-1,1)
df_7['Philippe Nord speed (km/h)']=Y_pred_philippe_nord.reshape(-1,1)
df_7['Philippe Nord speed (km/h) ground truth']=Y_test_philippe_nord.reshape(-1,1)
df_7['Magnan speed (km/h)']=Y_pred_magnan.reshape(-1,1)
df_7['Magnan speed (km/h) ground truth']=Y_test_magnan.reshape(-1,1)
df_7['Grinda speed (km/h)']=Y_pred_grinda.reshape(-1,1)
df_7['Grinda speed (km/h) ground truth']=Y_test_grinda.reshape(-1,1)
df_7



#20/1 from 7:00 to 8:54 (Monday)
df_1
#22/1 from 17:00 to 18:56 (Wednesday)
df_2
#24/01 from 17:00 to 18:56 (Friday)
df_3
#27/2 from 7:00 to 8:54 (Thursday)
df_4
#27/3 from 17:00 to 18:56 (Friday)
df_5
#22/07 from 17:00 to 18:56 (Wednesday)
df_6
#25/07 from 17:00 to 18:56 (Saturday)
df_7


writer = pd.ExcelWriter('/Users/nronzoni/Desktop/Voie Mathis/prediction/WALKFORWARD_SupportVectorRegression_prediction_speed_30min_NC.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_1.to_excel(writer, sheet_name='20-1-2020 morning')
df_2.to_excel(writer, sheet_name='22-1-2020 afternoon')
df_3.to_excel(writer, sheet_name='24-1-2020 afternoon')
df_4.to_excel(writer, sheet_name='27-2-2020 morning')
df_5.to_excel(writer, sheet_name='27-3-2020 afternoon')
df_6.to_excel(writer, sheet_name='22-7-2020 afternoon')
df_7.to_excel(writer, sheet_name='25-7-2020 afternoon')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

###### Plot 


df_no = pd.read_excel('/Users/nronzoni/Desktop/Voie Mathis/prediction/WALKFORWARD_SupportVectorRegression_prediction_speed_30min_NC.xlsx', sheet_name='22-7-2020 afternoon')

df_si = pd.read_excel('/Users/nronzoni/Desktop/Voie Mathis/prediction/WALKFORWARD_SupportVectorRegression_prediction_speed_30MIN.xlsx', sheet_name='22-07-2020 afternoon') 

#fix upper bound and lower bound for the flow 
df_si.min(axis=0)
df_no.min(axis=0)
df_si.max(axis=0)
df_no.max(axis=0)

minimum=5
maximum=85
#ground truth svr classifaction prediction 
cimiez_nord_si=df_si['Cimiez Nord speed (km/h)'].values
cimiez_nord_no=df_no['Cimiez Nord speed (km/h)'].values
cimiez_nord_GT=df_si['Cimiez Nord speed (km/h) ground truth'].values
philippe_nord_si=df_si['Philippe Nord speed (km/h)'].values
philippe_nord_no=df_no['Philippe Nord speed (km/h)'].values
philippe_nord_GT=df_si['Philippe Nord speed (km/h) ground truth'].values
magnan_si=df_si['Magnan speed (km/h)'].values
magnan_no=df_no['Magnan speed (km/h)'].values
magnan_GT=df_si['Magnan speed (km/h) ground truth'].values
grinda_si=df_si['Grinda speed (km/h)'].values
grinda_no=df_no['Grinda speed (km/h)'].values
grinda_GT=df_si['Grinda speed (km/h) ground truth'].values

x=index_third_period=pd.date_range('2014-02-10 17:00:00',periods=20, freq='6min')
len(x)
x=x.strftime("%H:%M")
              
plt.figure(figsize=(35,25))
plt.subplot(2,2,1)
plt.plot(x,cimiez_nord_si,'r-',label='Walk Forward prediction with clustering')
plt.plot(x,cimiez_nord_no,'b-',label='Walk Forward prediction')
plt.plot(x,cimiez_nord_GT,'g-',label='ground truth')
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((minimum,maximum))
plt.title('Cimiez Nord',fontsize=18)
plt.xticks(rotation=30,size=16)
plt.yticks(size=16)
plt.legend(loc='lower left',fontsize=18)
plt.subplot(2,2,2)
plt.plot(x,philippe_nord_si,'r-',label='Walk Forward prediction with clustering')
plt.plot(x,philippe_nord_no,'b-',label='Walk Forward prediction')
plt.plot(x,philippe_nord_GT,'g-',label='ground truth')
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((minimum,maximum))
plt.title('Philippe Nord',fontsize=18)
plt.xticks(rotation=30,size=16)
plt.yticks(size=16)
plt.legend(loc='lower left',fontsize=18)
plt.subplot(2,2,3)
plt.plot(x,magnan_si,'r-',label='Walk Forward prediction with clustering')
plt.plot(x,magnan_no,'b-',label='Walk Forward prediction')
plt.plot(x,magnan_GT,'g-',label='ground truth')
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((minimum,maximum))
plt.title('Magnan',fontsize=18)
plt.xticks(rotation=30,size=16)
plt.yticks(size=16)
plt.legend(loc='lower left',fontsize=18)
plt.subplot(2,2,4)
plt.plot(x,grinda_si,'r-',label='Walk Forward prediction with clustering')
plt.plot(x,grinda_no,'b-',label='Walk Forward prediction')
plt.plot(x,grinda_GT,'g-',label='ground truth')
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('km/h',labelpad=0,fontsize=18)
plt.ylim((minimum,maximum))
plt.title('Grinda',fontsize=18)
plt.xticks(rotation=30,size=16)
plt.yticks(size=16)
plt.legend(loc='upper left',fontsize=18)
plt.suptitle("22/7/2020 speed predictions: loops North direction Voie Mathis", fontsize=24, y=0.93)
plt.show()


