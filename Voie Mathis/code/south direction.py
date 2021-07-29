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

#IMPORT DATA South Direction 


augustin=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=0) 

carras_sortie=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=1)

carras_entre=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=2)

gloria=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=3)

philippe_sud=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=4)

cimiez_sud=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=5)

series=pd.Series(data=["speed"].values,index=augustin.index)

series.plot()
  
#philippe sud not contain speed value   
 
#SPEED
#augustin
augustin_speed=data_split_south(augustin)
## create daily time series train 
series_train_augustin_speed=daily_series(augustin_speed[0],170)
series_train_augustin_speed[0].shape
print('Min: %f, Max: %f' % (series_train_augustin_speed[1].data_min_, series_train_augustin_speed[1].data_max_))
#gloria
gloria_speed=data_split_south(gloria)
## create daily time series train 
series_train_gloria_speed=daily_series(gloria_speed[0],170)
series_train_gloria_speed[0].shape
print('Min: %f, Max: %f' % (series_train_gloria_speed[1].data_min_, series_train_gloria_speed[1].data_max_))
#cimiez_sud
cimiez_sud_speed=data_split_south(cimiez_sud)
## create daily time series train 
series_train_cimiez_sud_speed=daily_series(cimiez_sud_speed[0],170)
series_train_cimiez_sud_speed[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_sud_speed[1].data_min_, series_train_cimiez_sud_speed[1].data_max_))

#FLOW TRAIN
#augustin
augustin_flow=data_split_flow_south(augustin)
## create daily time series train 
series_train_augustin_flow=daily_series(augustin_flow[0],170)
series_train_augustin_flow[0].shape
print('Min: %f, Max: %f' % (series_train_augustin_flow[1].data_min_, series_train_augustin_flow[1].data_max_))
#carras sortie
carras_sortie_flow=data_split_flow_south_bis(carras_sortie)
## create daily time series train 
series_train_carras_sortie_flow=daily_series(carras_sortie_flow[0],170)
series_train_carras_sortie_flow[0].shape
print('Min: %f, Max: %f' % (series_train_carras_sortie_flow[1].data_min_, series_train_carras_sortie_flow[1].data_max_))
#carras entree
carras_entree_flow=data_split_flow_south_bis(carras_entre)
## create daily time series train 
series_train_carras_entree_flow=daily_series(carras_entree_flow[0],170)
series_train_carras_entree_flow[0].shape
print('Min: %f, Max: %f' % (series_train_carras_entree_flow[1].data_min_, series_train_carras_entree_flow[1].data_max_))
#gloria
gloria_flow=data_split_flow_south_bis(gloria)
## create daily time series train 
series_train_gloria_flow=daily_series(gloria_flow[0],170)
series_train_gloria_flow[0].shape
print('Min: %f, Max: %f' % (series_train_gloria_flow[1].data_min_, series_train_gloria_flow[1].data_max_))
#philippe sud 
philippe_sud_flow=data_split_flow_south_bis(philippe_sud)
## create daily time series train 
series_train_philippe_sud_flow=daily_series(philippe_sud_flow[0],170)
series_train_philippe_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_sud_flow[1].data_min_, series_train_philippe_sud_flow[1].data_max_))
#cimiez_sud
cimiez_sud_flow=data_split_flow_south_bis(cimiez_sud)
## create daily time series train 
series_train_cimiez_sud_flow=daily_series(cimiez_sud_flow[0],170)
series_train_cimiez_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_sud_flow[1].data_min_, series_train_cimiez_sud_flow[1].data_max_))


#multivariate time series train
multivariate=np.dstack((series_train_carras_sortie_flow[0],series_train_carras_entree_flow[0],series_train_gloria_flow[0],series_train_philippe_sud_flow[0],series_train_cimiez_sud_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#FLOW TEST
#augustin
## create daily time series test 
series_test_augustin_flow=daily_series(augustin_flow[1],170)
series_test_augustin_flow[0].shape
print('Min: %f, Max: %f' % (series_test_augustin_flow[1].data_min_, series_test_augustin_flow[1].data_max_))
#carras sortie
## create daily time series test 
series_test_carras_sortie_flow=daily_series(carras_sortie_flow[1],170)
series_test_carras_sortie_flow[0].shape
print('Min: %f, Max: %f' % (series_test_carras_sortie_flow[1].data_min_, series_test_carras_sortie_flow[1].data_max_))
#carras entree
carras_entree_flow=data_split_flow_south(carras_entre)
## create daily time series test 
series_test_carras_entree_flow=daily_series(carras_entree_flow[1],170)
series_test_carras_entree_flow[0].shape
print('Min: %f, Max: %f' % (series_test_carras_entree_flow[1].data_min_, series_test_carras_entree_flow[1].data_max_))
#gloria
## create daily time series test 
series_test_gloria_flow=daily_series(gloria_flow[1],170)
series_test_gloria_flow[0].shape
print('Min: %f, Max: %f' % (series_test_gloria_flow[1].data_min_, series_test_gloria_flow[1].data_max_))
#philippe sud 
## create daily time series test
series_test_philippe_sud_flow=daily_series(philippe_sud_flow[1],170)
series_test_philippe_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_test_philippe_sud_flow[1].data_min_, series_test_philippe_sud_flow[1].data_max_))
#cimiez_sud
## create daily time series test 
series_test_cimiez_sud_flow=daily_series(cimiez_sud_flow[1],170)
series_test_cimiez_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_test_cimiez_sud_flow[1].data_min_, series_test_cimiez_sud_flow[1].data_max_))


#multivariate time series test
multivariate_test=np.dstack((series_test_carras_sortie_flow[0],series_test_carras_entree_flow[0],series_test_gloria_flow[0],series_test_philippe_sud_flow[0],series_test_cimiez_sud_flow[0]))
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

#south direction 
#train
# from 1/1 to 13/1 
first_mid=pd.date_range('1/1/2019', periods=13, freq='D')
#from 15/1 to 16/1
second_mid=pd.date_range('1/15/2019', periods=2, freq='D')
#from 18/1 to  2/2
third_mid=pd.date_range('1/18/2019', periods=16, freq='D')
#from 15/3 to 8/4
fourth_mid=pd.date_range('3/15/2019', periods=25, freq='D')
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
#from 18/10 to 22/10
twelveth_mid=pd.date_range('10/18/2019', periods=5, freq='D')
# from 25/10 to 2/11
thirdteen_mid=pd.date_range('10/25/2019', periods=9, freq='D')
# from 4/11 to 12/11
fourthteen_mid=pd.date_range('11/4/2019', periods=9, freq='D')
# from 14/11 to 25/11
fifthteen_mid=pd.date_range('11/14/2019', periods=12, freq='D')
#from 27/11 to 8/12
sixtheen_mid=pd.date_range('11/27/2019', periods=12, freq='D')
#from 10/12 to 17/12
seventeen_mid=pd.date_range('12/10/2019', periods=8, freq='D')
#from 19/12 to 31/12 
eighteen_mid=pd.date_range('12/19/2019', periods=13, freq='D')

first_mid=pd.Series(data=first_mid)
second_mid=pd.Series(data=second_mid)
third_mid=pd.Series(data=third_mid)
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
fifthteen_mid=pd.Series(data=fifthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
index_train=pd.concat([first_mid,second_mid,third_mid,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,fifthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid],ignore_index=True)

#south direction bis  
#train
# from 1/1 to 13/1 
first_mid=pd.date_range('1/1/2019', periods=13, freq='D')
#from 15/1 to 16/1
second_mid=pd.date_range('1/15/2019', periods=2, freq='D')
#from 18/1 to  13/2
third_mid=pd.date_range('1/18/2019', periods=27, freq='D')
#from 16/2 to  18/2
third_mid_bis=pd.date_range('2/16/2019', periods=3, freq='D')
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
#from 18/10 to 22/10
twelveth_mid=pd.date_range('10/18/2019', periods=5, freq='D')
# from 25/10 to 2/11
thirdteen_mid=pd.date_range('10/25/2019', periods=9, freq='D')
# from 4/11 to 12/11
fourthteen_mid=pd.date_range('11/4/2019', periods=9, freq='D')
# from 14/11 to 25/11
fifthteen_mid=pd.date_range('11/14/2019', periods=12, freq='D')
#from 27/11 to 8/12
sixtheen_mid=pd.date_range('11/27/2019', periods=12, freq='D')
#from 10/12 to 17/12
seventeen_mid=pd.date_range('12/10/2019', periods=8, freq='D')
#from 19/12 to 31/12 
eighteen_mid=pd.date_range('12/19/2019', periods=13, freq='D')

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
fifthteen_mid=pd.Series(data=fifthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,fifthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid],ignore_index=True)
len(index_train)

#plot the result 
new=[]
for i in range(0,343):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
    
for i in range(0,343):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,343):
    if new[i] == 1:
        new[i] =0.05

for i in range(0,343):
    if new[i] ==4:
        new[i] =1
            
for i in range(0,343):
    if new[i] == 1:
        new[i] =4
        
for i in range(0,343):
    if new[i] == 2:
        new[i] =1

for i in range(0,343):
    if new[i] ==4:
        new[i] =2

#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis South Direction 2019 (train): débit de circulation loops and ramps', linewidth=2.3,dropzero=True,vmin=0) 

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
    if new[i] == 1:
        new[i] =0.05

for i in range(0,28):
    if new[i] ==4:
        new[i] =1

            
for i in range(0,28):
    if new[i] == 1:
        new[i] =4
        
for i in range(0,28):
    if new[i] == 2:
        new[i] =1

for i in range(0,28):
    if new[i] ==4:
        new[i] =2


events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis South Direction 2020 (test):débit de circulation loops and ramps', linewidth=2.3,dropzero=True,vmin=0) 


for i in range(0,35):
    if new[i] == 0.05:
        new[i]=0
    prediction_test[i]=new[i]

np.set_printoptions(threshold=400)        
prediction_test      





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

augustin_sample1=sample1[:,:,0]
augustin_sample1=series_train_augustin_flow[1].inverse_transform(augustin_sample1)
augustin_sample1.shape

gloria_sample1=sample1[:,:,1]
gloria_sample1=series_train_gloria_flow[1].inverse_transform(gloria_sample1)
gloria_sample1.shape

philippe_sud_sample1=sample1[:,:,2]
philippe_sud_sample1=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample1)
philippe_sud_sample1.shape

cimiez_sud_sample1=sample1[:,:,3]
cimiez_sud_sample1=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample1)
cimiez_sud_sample1.shape

####second cluster #######

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape


augustin_sample2=sample2[:,:,0]
augustin_sample2=series_train_augustin_flow[1].inverse_transform(augustin_sample2)
augustin_sample2.shape

gloria_sample2=sample2[:,:,1]
gloria_sample2=series_train_gloria_flow[1].inverse_transform(gloria_sample2)
gloria_sample2.shape

philippe_sud_sample2=sample1[:,:,2]
philippe_sud_sample2=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample2)
philippe_sud_sample2.shape

cimiez_sud_sample2=sample2[:,:,3]
cimiez_sud_sample2=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample2)
cimiez_sud_sample2.shape

#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape


augustin_sample3=sample3[:,:,0]
augustin_sample3=series_train_augustin_flow[1].inverse_transform(augustin_sample3)
augustin_sample3.shape

gloria_sample3=sample3[:,:,1]
gloria_sample3=series_train_gloria_flow[1].inverse_transform(gloria_sample3)
gloria_sample3.shape

philippe_sud_sample3=sample3[:,:,2]
philippe_sud_sample3=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample3)
philippe_sud_sample3.shape

cimiez_sud_sample3=sample3[:,:,3]
cimiez_sud_sample3=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample3)
cimiez_sud_sample3.shape
#select randomly time series from fourth cluster 
cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:20]

sample4.shape


augustin_sample4=sample4[:,:,0]
augustin_sample4=series_train_augustin_flow[1].inverse_transform(augustin_sample4)
augustin_sample4.shape

gloria_sample4=sample4[:,:,1]
gloria_sample4=series_train_gloria_flow[1].inverse_transform(gloria_sample4)
gloria_sample4.shape

philippe_sud_sample4=sample4[:,:,2]
philippe_sud_sample4=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample4)
philippe_sud_sample4.shape

cimiez_sud_sample4=sample4[:,:,3]
cimiez_sud_sample4=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample4)
cimiez_sud_sample4.shape
# plot the centroids k=4 
centroids=km_dba.cluster_centers_
centroids.shape

#k=0#
augustin_1=centroids[0][:,0]
augustin_1=augustin_1.reshape((len(augustin_1), 1))

gloria_1=centroids[0][:,1]
gloria_1=gloria_1.reshape((len(gloria_1), 1))

philippe_sud_1=centroids[0][:,2]
philippe_sud_1=philippe_sud_1.reshape((len(philippe_sud_1), 1))

cimiez_sud_1=centroids[0][:,3]
cimiez_sud_1=cimiez_sud_1.reshape((len(cimiez_sud_1), 1))


#k=1#
augustin_2=centroids[1][:,0]
augustin_2=augustin_2.reshape((len(augustin_2), 1))

gloria_2=centroids[1][:,1]
gloria_2=gloria_2.reshape((len(gloria_2), 1))

philippe_sud_2=centroids[1][:,2]
philippe_sud_2=philippe_sud_2.reshape((len(philippe_sud_2), 1))

cimiez_sud_2=centroids[1][:,3]
cimiez_sud_2=cimiez_sud_2.reshape((len(cimiez_sud_2), 1))

#k=2#
augustin_3=centroids[2][:,0]
augustin_3=augustin_3.reshape((len(augustin_3), 1))

gloria_3=centroids[2][:,1]
gloria_3=gloria_3.reshape((len(gloria_3), 1))

philippe_sud_3=centroids[2][:,2]
philippe_sud_3=philippe_sud_3.reshape((len(philippe_sud_3), 1))

cimiez_sud_3=centroids[2][:,3]
cimiez_sud_3=cimiez_sud_3.reshape((len(cimiez_sud_3), 1))
#k=3#
augustin_4=centroids[3][:,0]
augustin_4=augustin_4.reshape((len(augustin_4), 1))

gloria_4=centroids[3][:,1]
gloria_4=gloria_4.reshape((len(gloria_4), 1))

philippe_sud_4=centroids[3][:,2]
philippe_sud_4=philippe_sud_4.reshape((len(philippe_sud_4), 1))

cimiez_sud_4=centroids[3][:,3]
cimiez_sud_4=cimiez_sud_4.reshape((len(cimiez_sud_4), 1))

import matplotlib.pyplot as plt
fig = plt.gcf()

x=np.arange(6,23,0.1)
len(x)


plt.figure(figsize=(35,40))
plt.subplot(4,4,1)
for i in range(0,20):
    plt.plot(x,augustin_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_augustin_flow[1].inverse_transform(augustin_1),'#33cc33', label = 'Augustin',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,4,2)
for i in range(0,20):
    plt.plot(x,gloria_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_1),'#33cc33', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,4,3)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_1),'#33cc33', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,4,4)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_1),'#33cc33', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(4,4,5)
for i in range(0,20):
    plt.plot(x,augustin_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_augustin_flow[1].inverse_transform(augustin_2),'#ff9900', label = 'Augustin',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,4,6)
for i in range(0,20):
    plt.plot(x,gloria_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_2),'#ff9900', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,4,7)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_2),'#ff9900', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,4,8)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_2),'#ff9900', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(4,4,9)
for i in range(0,20):
    plt.plot(x,augustin_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_augustin_flow[1].inverse_transform(augustin_3),'#ff0066', label = 'Augustin',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,4,10)
for i in range(0,20):
    plt.plot(x,gloria_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_3),'#ff0066', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,4,11)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_3),'#ff0066', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,4,12)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_3),'#ff0066', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(4,4,13)
for i in range(0,20):
    plt.plot(x,augustin_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_augustin_flow[1].inverse_transform(augustin_4),'#476b6b', label = 'Augustin',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,4,14)
for i in range(0,20):
    plt.plot(x,gloria_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_4),'#476b6b', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,4,15)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_4),'#476b6b', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((0,500))
plt.title('k=3')
plt.legend(loc='upper right')
plt.subplot(4,4,16)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample4[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_4),'#476b6b', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('veh',labelpad=0)
plt.ylim((0,500))
plt.title('k=3')
plt.legend(loc='upper right')
plt.figtext(0.5,0.30, "No working days January-August", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.50, "No working days September-December ", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.70, "Working days September-December", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.90, "Working days January-August", ha="center", va="top", fontsize=14, color="r")
plt.show()


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

carras_sortie_sample1=sample1[:,:,0]
carras_sortie_sample1=series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_sample1)
carras_sortie_sample1.shape

carras_entree_sample1=sample1[:,:,1]
carras_entree_sample1=series_train_carras_entree_flow[1].inverse_transform(carras_entree_sample1)
carras_entree_sample1.shape

gloria_sample1=sample1[:,:,2]
gloria_sample1=series_train_gloria_flow[1].inverse_transform(gloria_sample1)
gloria_sample1.shape

philippe_sud_sample1=sample1[:,:,3]
philippe_sud_sample1=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample1)
philippe_sud_sample1.shape

cimiez_sud_sample1=sample1[:,:,4]
cimiez_sud_sample1=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample1)
cimiez_sud_sample1.shape


####second cluster #######

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape

carras_sortie_sample2=sample2[:,:,0]
carras_sortie_sample2=series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_sample2)
carras_sortie_sample2.shape

carras_entree_sample2=sample2[:,:,1]
carras_entree_sample2=series_train_carras_entree_flow[1].inverse_transform(carras_entree_sample2)
carras_entree_sample2.shape

gloria_sample2=sample2[:,:,2]
gloria_sample2=series_train_gloria_flow[1].inverse_transform(gloria_sample2)
gloria_sample2.shape

philippe_sud_sample2=sample2[:,:,3]
philippe_sud_sample2=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample2)
philippe_sud_sample2.shape

cimiez_sud_sample2=sample2[:,:,4]
cimiez_sud_sample2=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample2)
cimiez_sud_sample2.shape


#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape

carras_sortie_sample3=sample3[:,:,0]
carras_sortie_sample3=series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_sample3)
carras_sortie_sample3.shape

carras_entree_sample3=sample3[:,:,1]
carras_entree_sample3=series_train_carras_entree_flow[1].inverse_transform(carras_entree_sample3)
carras_entree_sample3.shape

gloria_sample3=sample3[:,:,2]
gloria_sample3=series_train_gloria_flow[1].inverse_transform(gloria_sample3)
gloria_sample3.shape

philippe_sud_sample3=sample3[:,:,3]
philippe_sud_sample3=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample3)
philippe_sud_sample3.shape

cimiez_sud_sample3=sample3[:,:,4]
cimiez_sud_sample3=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample3)
cimiez_sud_sample3.shape


#k=0#
carras_sortie_1=centroids[0][:,0]
carras_sortie_1=carras_sortie_1.reshape((len(carras_sortie_1), 1))

carras_entree_1=centroids[0][:,1]
carras_entree_1=carras_entree_1.reshape((len(carras_entree_1), 1))

gloria_1=centroids[0][:,2]
gloria_1=gloria_1.reshape((len(gloria_1), 1))

philippe_sud_1=centroids[0][:,3]
philippe_sud_1=philippe_sud_1.reshape((len(philippe_sud_1), 1))

cimiez_sud_1=centroids[0][:,4]
cimiez_sud_1=cimiez_sud_1.reshape((len(cimiez_sud_1), 1))


#k=1#
carras_sortie_2=centroids[1][:,0]
carras_sortie_2=carras_sortie_2.reshape((len(carras_sortie_2), 1))

carras_entree_2=centroids[1][:,1]
carras_entree_2=carras_entree_2.reshape((len(carras_entree_2), 1))

gloria_2=centroids[1][:,2]
gloria_2=gloria_2.reshape((len(gloria_2), 1))

philippe_sud_2=centroids[1][:,3]
philippe_sud_2=philippe_sud_2.reshape((len(philippe_sud_2), 1))

cimiez_sud_2=centroids[1][:,4]
cimiez_sud_2=cimiez_sud_2.reshape((len(cimiez_sud_2), 1))

#k=2#
carras_sortie_3=centroids[2][:,0]
carras_sortie_3=carras_sortie_3.reshape((len(carras_sortie_3), 1))

carras_entree_3=centroids[2][:,1]
carras_entree_3=carras_entree_3.reshape((len(carras_entree_3), 1))

gloria_3=centroids[2][:,2]
gloria_3=gloria_3.reshape((len(gloria_3), 1))

philippe_sud_3=centroids[2][:,3]
philippe_sud_3=philippe_sud_3.reshape((len(philippe_sud_3), 1))

cimiez_sud_3=centroids[2][:,4]
cimiez_sud_3=cimiez_sud_3.reshape((len(cimiez_sud_3), 1))



x=np.arange(6,23,0.1)
len(x)


plt.figure(figsize=(35,35))
plt.subplot(3,5,1)
for i in range(0,20):
    plt.plot(x,carras_sortie_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_2),'#33cc33', label = 'Carras sortie',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,2)
for i in range(0,20):
    plt.plot(x,carras_entree_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_entree_flow[1].inverse_transform(carras_entree_2),'#33cc33', label = 'Carras entrèe',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,3)
for i in range(0,20):
    plt.plot(x,gloria_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_2),'#33cc33', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,4)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_2),'#33cc33', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,5)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_2),'#33cc33', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('débit de circulation veh',labelpad=0)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=0',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,6)
for i in range(0,20):
    plt.plot(x,carras_sortie_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_1),'#0033cc', label = 'Carras sortie',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,7)
for i in range(0,20):
    plt.plot(x,carras_entree_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_entree_flow[1].inverse_transform(carras_entree_1),'#0033cc', label = 'Carras entrèe',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,8)
for i in range(0,20):
    plt.plot(x,gloria_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_1),'#0033cc', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,9)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_1),'#0033cc', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,10)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_1),'#0033cc', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=1',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,11)
for i in range(0,20):
    plt.plot(x,carras_sortie_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_sortie_flow[1].inverse_transform(carras_sortie_3),'#666699', label = 'Carras sortie',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,12)
for i in range(0,20):
    plt.plot(x,carras_entree_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_carras_entree_flow[1].inverse_transform(carras_entree_3),'#666699', label = 'Carras entrèe',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,250))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,13)
for i in range(0,20):
    plt.plot(x,gloria_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_3),'#666699', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,14)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_3),'#666699', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.subplot(3,5,15)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_3),'#666699', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day',fontsize=18)
plt.ylabel('débit de circulation veh',labelpad=0,fontsize=18)
plt.ylim((0,450))
plt.xticks(size=16)
plt.yticks(size=16)
plt.title('k=2',fontsize=18)
plt.legend(loc='upper right',fontsize=18)
plt.figtext(0.5,0.38, "Working days in winter", ha="center", va="top", fontsize=24, color="r")
plt.figtext(0.5,0.65, "Working days in summer", ha="center", va="top", fontsize=24, color="r")
plt.figtext(0.5,0.90, "No Working days", ha="center", va="top", fontsize=24, color="r")
plt.show()

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


gloria_sample1=sample1[:,:,0]
gloria_sample1=series_train_gloria_flow[1].inverse_transform(gloria_sample1)
gloria_sample1.shape

philippe_sud_sample1=sample1[:,:,1]
philippe_sud_sample1=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample1)
philippe_sud_sample1.shape

cimiez_sud_sample1=sample1[:,:,2]
cimiez_sud_sample1=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample1)
cimiez_sud_sample1.shape


####second cluster #######

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:20]

sample2.shape

gloria_sample2=sample2[:,:,0]
gloria_sample2=series_train_gloria_flow[1].inverse_transform(gloria_sample2)
gloria_sample2.shape

philippe_sud_sample2=sample2[:,:,1]
philippe_sud_sample2=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample2)
philippe_sud_sample2.shape

cimiez_sud_sample2=sample2[:,:,2]
cimiez_sud_sample2=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample2)
cimiez_sud_sample2.shape


#select randomly time series from third cluster 
cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:20]

sample3.shape


gloria_sample3=sample3[:,:,0]
gloria_sample3=series_train_gloria_flow[1].inverse_transform(gloria_sample3)
gloria_sample3.shape

philippe_sud_sample3=sample3[:,:,1]
philippe_sud_sample3=series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_sample3)
philippe_sud_sample3.shape

cimiez_sud_sample3=sample3[:,:,2]
cimiez_sud_sample3=series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_sample3)
cimiez_sud_sample3.shape


#k=0#

gloria_1=centroids[0][:,0]
gloria_1=gloria_1.reshape((len(gloria_1), 1))

philippe_sud_1=centroids[0][:,1]
philippe_sud_1=philippe_sud_1.reshape((len(philippe_sud_1), 1))

cimiez_sud_1=centroids[0][:,2]
cimiez_sud_1=cimiez_sud_1.reshape((len(cimiez_sud_1), 1))


#k=1#

gloria_2=centroids[1][:,0]
gloria_2=gloria_2.reshape((len(gloria_2), 1))

philippe_sud_2=centroids[1][:,1]
philippe_sud_2=philippe_sud_2.reshape((len(philippe_sud_2), 1))

cimiez_sud_2=centroids[1][:,2]
cimiez_sud_2=cimiez_sud_2.reshape((len(cimiez_sud_2), 1))

#k=2#

gloria_3=centroids[2][:,0]
gloria_3=gloria_3.reshape((len(gloria_3), 1))

philippe_sud_3=centroids[2][:,1]
philippe_sud_3=philippe_sud_3.reshape((len(philippe_sud_3), 1))

cimiez_sud_3=centroids[2][:,2]
cimiez_sud_3=cimiez_sud_3.reshape((len(cimiez_sud_3), 1))



x=np.arange(6,23,0.1)
len(x)


plt.figure(figsize=(35,35))
plt.subplot(3,3,1)
for i in range(0,20):
    plt.plot(x,gloria_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_2),'#33cc33', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(3,3,2)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_2),'#33cc33', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(3,3,3)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample2[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_2),'#33cc33', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=0')
plt.legend(loc='upper right')
plt.subplot(3,3,4)
for i in range(0,20):
    plt.plot(x,gloria_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_3),'#0033cc', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(3,3,5)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_3),'#0033cc', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(3,3,6)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample3[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_3),'#0033cc', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=1')
plt.legend(loc='upper right')
plt.subplot(3,3,7)
for i in range(0,20):
    plt.plot(x,gloria_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_gloria_flow[1].inverse_transform(gloria_1),'#666699', label = 'Gloria',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(3,3,8)
for i in range(0,20):
    plt.plot(x,philippe_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_philippe_sud_flow[1].inverse_transform(philippe_sud_1),'#666699', label = 'Philippe Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=2')
plt.legend(loc='upper right')
plt.subplot(3,3,9)
for i in range(0,20):
    plt.plot(x,cimiez_sud_sample1[i],'k-', alpha=.1)
plt.plot(x,series_train_cimiez_sud_flow[1].inverse_transform(cimiez_sud_1),'#666699', label = 'Cimiez Sud',linewidth=3)
plt.xlabel('hours of the day')
plt.ylabel('occupancy rate %',labelpad=0)
plt.ylim((0,35))
plt.title('k=2')
plt.legend(loc='upper right')
plt.figtext(0.5,0.36, "Working days higher traffic peak in the morning", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.63, "Working days higher traffic peak in the afternoon", ha="center", va="top", fontsize=14, color="r")
plt.figtext(0.5,0.90, "No Working days, school breaks", ha="center", va="top", fontsize=14, color="r")
plt.show()




