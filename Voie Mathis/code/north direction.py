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
#SPEED
#cimiez nord
cimiez_nord_speed=data_split_north(cimiez_nord)
## create daily time series train 
series_train_cimiez_nord_speed=daily_series(cimiez_nord_speed[0],170)
series_train_cimiez_nord_speed[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_nord_speed[1].data_min_, series_train_cimiez_nord_speed[1].data_max_))
#philippe nord
philippe_nord_speed=data_split_north(philippe_nord)
## create daily time series train 
series_train_philippe_nord_speed=daily_series(philippe_nord_speed[0],170)
series_train_philippe_nord_speed[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_nord_speed[1].data_min_, series_train_philippe_nord_speed[1].data_max_))
#magnan
magnan_speed=data_split_north(magnan)
## create daily time series train 
series_train_magnan_speed=daily_series(magnan_speed[0],170)
series_train_magnan_speed[0].shape
print('Min: %f, Max: %f' % (series_train_magnan_speed[1].data_min_, series_train_magnan_speed[1].data_max_))
#grinda
grinda_speed=data_split_north(grinda)
## create daily time series train 
series_train_grinda_speed=daily_series(grinda_speed[0],170)
series_train_grinda_speed[0].shape
print('Min: %f, Max: %f' % (series_train_grinda_speed[1].data_min_, series_train_grinda_speed[1].data_max_))

#FLOW
#cimiez nord
cimiez_nord_flow=data_split_flow_north_bis(cimiez_nord)
## create daily time series train 
series_train_cimiez_nord_flow=daily_series(cimiez_nord_flow[0],170)
series_train_cimiez_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_nord_flow[1].data_min_, series_train_cimiez_nord_flow[1].data_max_))
#philippe nord
philippe_nord_flow=data_split_flow_north_bis(philippe_nord)
## create daily time series train 
series_train_philippe_nord_flow=daily_series(philippe_nord_flow[0],170)
series_train_philippe_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_nord_flow[1].data_min_, series_train_philippe_nord_flow[1].data_max_))
#magnan
magnan_flow=data_split_flow_north_bis(magnan)
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
multivariate=np.dstack((series_train_cimiez_nord_flow[0],series_train_philippe_nord_flow[0],series_train_magnan_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#FLOW
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
multivariate_test=np.dstack((series_test_cimiez_nord_flow[0],series_test_philippe_nord_flow[0],series_test_magnan_flow[0]))
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
for i in range(0,342):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
    
for i in range(0,342):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,342):
    if new[i] == 2:
        new[i] =0.05

for i in range(0,342):
    if new[i] ==4:
        new[i] =2

#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis North Direction 2019 (train):débit de circulation loops', linewidth=2.3,dropzero=True,vmin=0) 

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
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis North Direction 2020 (test):débit de circulation loops', linewidth=2.3,dropzero=True,vmin=0) 


for i in range(0,35):
    if new[i] == 0.05:
        new[i]=0
    prediction_test[i]=new[i]

np.set_printoptions(threshold=400)        
prediction_test      




#centroids 
centroids=km_dba.cluster_centers_

centroids.shape

#dataframe to select days which belogns to the closest cluster 
len(index_train)
columns=['days','k']
index=np.arange(357)
len(index)
dataframe_train=pd.DataFrame(columns=columns,index=index)
dataframe_train['days']=index_train
dataframe_train['k']=prediction_train
dataframe_train
dataframe_train['day'] = dataframe_train['days'].dt.day
dataframe_train['month'] =dataframe_train['days'].dt.month
dataframe_train['year'] = dataframe_train['days'].dt.year
dataframe_train.drop(['days'], axis=1)
dataframe_train = dataframe_train[['year', 'month', 'day', 'k']]
dataframe_train
dataframe_train.to_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction with ramps/Classification of the days.xlsx')
#if k=0 
days_cluster=dataframe_train[dataframe_train['k']==2].index
len(days_cluster)
#multivariate time series train speed
#check if multivariate_speed already exist
multivariate_speed=np.dstack((series_train_S54_speed[0],series_train_S1706_speed[0],series_train_S56_speed[0],series_train_S57_speed[0],series_train_S1707_speed[0],series_train_S59_speed[0],series_train_S60_speed[0],series_train_S61_speed[0]))
multivariate_time_series_train_speed = to_time_series(multivariate_speed)
print(multivariate_time_series_train_speed.shape)

#multivariate time series test speed
multivariate_test_speed=np.dstack((series_test_S54_speed[0],series_test_S1706_speed[0],series_test_S56_speed[0],series_test_S57_speed[0],series_test_S1707_speed[0],series_test_S59_speed[0],series_test_S60_speed[0],series_test_S61_speed[0]))
multivariate_time_series_test_speed = to_time_series(multivariate_test_speed)
print(multivariate_time_series_test_speed.shape)

pd.set_option('display.max_seq_items', 200)
print(days_cluster)
multivariate_time_series_train_speed_subset=multivariate_time_series_train_speed[(17,  38,  45,  59,  66,  72,  80,  87,  94, 101, 115, 122, 129,
            136, 142, 143, 150, 156, 157, 163, 164, 169, 170, 172, 173, 174,
            179, 187, 188, 192, 193, 194, 195, 200, 201, 202, 207, 208, 209,
            214, 215, 216, 220, 221, 222, 223, 229, 230, 233, 234, 235, 242,
            247, 254, 260, 261, 268, 275, 282, 289, 296, 303, 310, 317, 322,
            331, 338, 345),:,:]

multivariate_time_series_train_speed_subset.shape
#day nearest to the cluster centroid 
closest(multivariate_time_series_train,prediction_train,centroids,3,events_train)

############# plot of the centroids 
########## centroids#########################
########################################### k=4 ################################

##### first cluster #######
cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:20]

sample1.shape

S54_sample1=sample1[:,:,0]
S54_sample1=series_train_S54_flow[1].inverse_transform(S54_sample1)
S54_sample1.shape

S1706_sample1=sample1[:,:,1]
S1706_sample1=series_train_S1706_flow[1].inverse_transform(S1706_sample1)
S1706_sample1.shape

S56_sample1=sample1[:,:,2]
S56_sample1=series_train_S56_flow[1].inverse_transform(S56_sample1)
S56_sample1.shape

S57_sample1=sample1[:,:,3]
S57_sample1=series_train_S57_flow[1].inverse_transform(S57_sample1)
S57_sample1.shape

S1707_sample1=sample1[:,:,4]
S1707_sample1=series_train_S1707_flow[1].inverse_transform(S1707_sample1)
S1707_sample1.shape

S59_sample1=sample1[:,:,5]
S59_sample1=series_train_S59_flow[1].inverse_transform(S59_sample1)
S59_sample1.shape

S60_sample1=sample1[:,:,6]
S60_sample1=series_train_S60_flow[1].inverse_transform(S60_sample1)
S60_sample1.shape

S61_sample1=sample1[:,:,7]
S61_sample1=series_train_S61_flow[1].inverse_transform(S61_sample1)
S61_sample1.shape


####second cluster #######

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape

S54_sample2=sample2[:,:,0]
S54_sample2=series_train_S54_flow[1].inverse_transform(S54_sample2)
S54_sample2.shape

S1706_sample2=sample2[:,:,1]
S1706_sample2=series_train_S1706_flow[1].inverse_transform(S1706_sample2)
S1706_sample2.shape

S56_sample2=sample2[:,:,2]
S56_sample2=series_train_S56_flow[1].inverse_transform(S56_sample2)
S56_sample2.shape

S57_sample2=sample2[:,:,3]
S57_sample2=series_train_S57_flow[1].inverse_transform(S57_sample2)
S57_sample2.shape

S1707_sample2=sample2[:,:,4]
S1707_sample2=series_train_S1707_flow[1].inverse_transform(S1707_sample2)
S1707_sample2.shape

S59_sample2=sample2[:,:,5]
S59_sample2=series_train_S59_flow[1].inverse_transform(S59_sample2)
S59_sample2.shape

S60_sample2=sample2[:,:,6]
S60_sample2=series_train_S60_flow[1].inverse_transform(S60_sample2)
S60_sample2.shape

S61_sample2=sample2[:,:,7]
S61_sample2=series_train_S61_flow[1].inverse_transform(S61_sample2)
S61_sample2.shape


#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape


S54_sample3=sample3[:,:,0]
S54_sample3=series_train_S54_flow[1].inverse_transform(S54_sample3)
S54_sample3.shape

S1706_sample3=sample3[:,:,1]
S1706_sample3=series_train_S1706_flow[1].inverse_transform(S1706_sample3)
S1706_sample3.shape

S56_sample3=sample3[:,:,2]
S56_sample3=series_train_S56_flow[1].inverse_transform(S56_sample3)
S56_sample3.shape

S57_sample3=sample3[:,:,3]
S57_sample3=series_train_S57_flow[1].inverse_transform(S57_sample3)
S57_sample3.shape

S1707_sample3=sample3[:,:,4]
S1707_sample3=series_train_S1707_flow[1].inverse_transform(S1707_sample3)
S1707_sample3.shape

S59_sample3=sample3[:,:,5]
S59_sample3=series_train_S59_flow[1].inverse_transform(S59_sample3)
S59_sample3.shape

S60_sample3=sample3[:,:,6]
S60_sample3=series_train_S60_flow[1].inverse_transform(S60_sample3)
S60_sample3.shape

S61_sample3=sample3[:,:,7]
S61_sample3=series_train_S61_flow[1].inverse_transform(S61_sample3)
S61_sample3.shape

#select randomly time series from fourth cluster 
cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:20]

sample4.shape

S54_sample4=sample4[:,:,0]
S54_sample4=series_train_S54_flow[1].inverse_transform(S54_sample4)
S54_sample4.shape

S1706_sample4=sample4[:,:,1]
S1706_sample4=series_train_S1706_flow[1].inverse_transform(S1706_sample4)
S1706_sample4.shape

S56_sample4=sample4[:,:,2]
S56_sample4=series_train_S56_flow[1].inverse_transform(S56_sample4)
S56_sample4.shape

S57_sample4=sample4[:,:,3]
S57_sample4=series_train_S57_flow[1].inverse_transform(S57_sample4)
S57_sample4.shape

S1707_sample4=sample4[:,:,4]
S1707_sample4=series_train_S57_flow[1].inverse_transform(S1707_sample4)
S1707_sample4.shape

S59_sample4=sample4[:,:,5]
S59_sample4=series_train_S59_flow[1].inverse_transform(S59_sample4)
S59_sample4.shape

S60_sample4=sample4[:,:,6]
S60_sample4=series_train_S60_flow[1].inverse_transform(S60_sample4)
S60_sample4.shape

S61_sample4=sample4[:,:,7]
S61_sample4=series_train_S61_flow[1].inverse_transform(S61_sample4)
S61_sample4.shape


# plot the centroids k=4 
centroids=km_dba.cluster_centers_
centroids.shape

#k=0#
S54_1=centroids[0][:,0]
S54_1=S54_1.reshape((len(S54_1), 1))

S1706_1=centroids[0][:,1]
S1706_1=S1706_1.reshape((len(S1706_1), 1))

S56_1=centroids[0][:,2]
S56_1=S56_1.reshape((len(S56_1), 1))

S57_1=centroids[0][:,3]
S57_1=S57_1.reshape((len(S57_1), 1))

S1707_1=centroids[0][:,4]
S1707_1=S1707_1.reshape((len(S1707_1), 1))

S59_1=centroids[0][:,5]
S59_1=S59_1.reshape((len(S59_1), 1))

S60_1=centroids[0][:,6]
S60_1=S60_1.reshape((len(S60_1), 1))

S61_1=centroids[0][:,7]
S61_1=S61_1.reshape((len(S61_1), 1))

#k=1#
S54_2=centroids[1][:,0]
S54_2=S54_2.reshape((len(S54_2), 1))

S1706_2=centroids[1][:,1]
S1706_2=S1706_2.reshape((len(S1706_2), 1))

S56_2=centroids[1][:,2]
S56_2=S56_2.reshape((len(S56_2), 1))

S57_2=centroids[1][:,3]
S57_2=S57_2.reshape((len(S57_2), 1))

S1707_2=centroids[1][:,4]
S1707_2=S1707_2.reshape((len(S1707_2), 1))

S59_2=centroids[1][:,5]
S59_2=S59_2.reshape((len(S59_2), 1))

S60_2=centroids[1][:,6]
S60_2=S60_2.reshape((len(S60_2), 1))

S61_2=centroids[1][:,7]
S61_2=S61_2.reshape((len(S61_2), 1))

#k=2#
S54_3=centroids[2][:,0]
S54_3=S54_3.reshape((len(S54_3), 1))

S1706_3=centroids[2][:,1]
S1706_3=S1706_3.reshape((len(S1706_3), 1))

S56_3=centroids[2][:,2]
S56_3=S56_3.reshape((len(S56_3), 1))

S57_3=centroids[2][:,3]
S57_3=S57_3.reshape((len(S57_3), 1))

S1707_3=centroids[2][:,4]
S1707_3=S1707_3.reshape((len(S1707_3), 1))

S59_3=centroids[2][:,5]
S59_3=S59_3.reshape((len(S59_3), 1))

S60_3=centroids[2][:,6]
S60_3=S60_3.reshape((len(S60_3), 1))

S61_3=centroids[2][:,7]
S61_3=S61_3.reshape((len(S61_3), 1))

#k=3#
S54_4=centroids[3][:,0]
S54_4=S54_4.reshape((len(S54_4), 1))

S1706_4=centroids[3][:,1]
S1706_4=S1706_4.reshape((len(S1706_4), 1))

S56_4=centroids[3][:,2]
S56_4=S56_4.reshape((len(S56_4), 1))

S57_4=centroids[3][:,3]
S57_4=S57_4.reshape((len(S57_4), 1))

S1707_4=centroids[3][:,4]
S1707_4=S1707_4.reshape((len(S1707_4), 1))

S59_4=centroids[3][:,5]
S59_4=S59_4.reshape((len(S59_4), 1))

S60_4=centroids[3][:,6]
S60_4=S60_4.reshape((len(S60_4), 1))

S61_4=centroids[3][:,7]
S61_4=S61_4.reshape((len(S61_4), 1))

import matplotlib.pyplot as plt
fig = plt.gcf()

x=np.arange(5,23,0.1)
len(x)


plt.figure(figsize=(35,40))
plt.subplot(4,8,1)
for i in range(0,20):
    plt.plot(x,S54_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_3),'#33cc33', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,2)
for i in range(0,20):
    plt.plot(x,S1706_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_3),'#33cc33', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,3)
for i in range(0,20):
    plt.plot(x,S56_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_3),'#33cc33', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,4)
for i in range(0,20):
    plt.plot(x,S57_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_3),'#33cc33', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,5)
for i in range(0,20):
    plt.plot(x,S1707_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_3),'#33cc33', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,6)
for i in range(0,20):
    plt.plot(x,S59_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_3),'#33cc33', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,7)
for i in range(0,20):
    plt.plot(x,S60_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_3),'#33cc33', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,8)
for i in range(0,20):
    plt.plot(x,S61_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_3),'#33cc33', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,8,9)
for i in range(0,20):
    plt.plot(x,S54_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_2),'#ff9900', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,10)
for i in range(0,20):
    plt.plot(x,S1706_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_2),'#ff9900', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,11)
for i in range(0,20):
    plt.plot(x,S56_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_2),'#ff9900', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,12)
for i in range(0,20):
    plt.plot(x,S57_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_2),'#ff9900', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,13)
for i in range(0,20):
    plt.plot(x,S1707_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_2),'#ff9900', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,14)
for i in range(0,20):
    plt.plot(x,S59_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_2),'#ff9900', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,15)
for i in range(0,20):
    plt.plot(x,S60_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_2),'#ff9900', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,16)
for i in range(0,20):
    plt.plot(x,S61_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_2),'#ff9900', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,8,17)
for i in range(0,20):
    plt.plot(x,S54_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_1),'#ff0066', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,18)
for i in range(0,20):
    plt.plot(x,S1706_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_1),'#ff0066', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,19)
for i in range(0,20):
    plt.plot(x,S56_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_1),'#ff0066', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,20)
for i in range(0,20):
    plt.plot(x,S57_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_1),'#ff0066', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,21)
for i in range(0,20):
    plt.plot(x,S1707_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_1),'#ff0066', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,22)
for i in range(0,20):
    plt.plot(x,S59_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_1),'#ff0066', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,23)
for i in range(0,20):
    plt.plot(x,S60_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_1),'#ff0066', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,24)
for i in range(0,20):
    plt.plot(x,S61_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_1),'#ff0066', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,8,25)
for i in range(0,20):
    plt.plot(x,S54_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S54_flow[1].inverse_transform(S54_4),'#476b6b', label = 'S54',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,26)
for i in range(0,20):
    plt.plot(x,S1706_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S1706_flow[1].inverse_transform(S1706_4),'#476b6b', label = 'S1706',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,27)
for i in range(0,20):
    plt.plot(x,S56_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S56_flow[1].inverse_transform(S56_4),'#476b6b', label = 'S56',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,28)
for i in range(0,20):
    plt.plot(x,S57_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S57_flow[1].inverse_transform(S57_4),'#476b6b', label = 'S57',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,29)
for i in range(0,20):
    plt.plot(x,S1707_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S1707_flow[1].inverse_transform(S1707_4),'#476b6b', label = 'S1707',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,30)
for i in range(0,20):
    plt.plot(x,S59_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S59_flow[1].inverse_transform(S59_4),'#476b6b', label = 'S59',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,31)
for i in range(0,20):
    plt.plot(x,S60_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S60_flow[1].inverse_transform(S60_4),'#476b6b', label = 'S60',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,8,32)
for i in range(0,20):
    plt.plot(x,S61_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_S61_flow[1].inverse_transform(S61_4),'#476b6b', label = 'S61',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,9800))
plt.title('k=3')
plt.legend(loc='upper right')
plt.figtext(0.5,0.30, "January,February and December", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.50, "Fridays, Wednesdays and Thursdays in June, July and August", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.70, "No working days", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.90, "First part of the week from May to November", ha="center", va="top", fontsize=14, color="r")
plt.show()


