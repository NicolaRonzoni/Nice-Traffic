#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:48:23 2021

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

#IMPORT DATA South Direction 

augustin=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=0) 

carras_sortie=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=1)

carras_entre=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=2)

gloria=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=3)

philippe_sud=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=4)

cimiez_sud=pd.read_excel(r"Voie Mathis South direction.xlsx",sheet_name=5)


#TRAIN
#cimiez nord
cimiez_nord_flow=data_split_flow_7loops(cimiez_nord)
## create daily time series train 
series_train_cimiez_nord_flow=daily_series(cimiez_nord_flow[0],170)
series_train_cimiez_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_nord_flow[1].data_min_, series_train_cimiez_nord_flow[1].data_max_))
#philippe nord
philippe_nord_flow=data_split_flow_7loops(philippe_nord)
## create daily time series train 
series_train_philippe_nord_flow=daily_series(philippe_nord_flow[0],170)
series_train_philippe_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_nord_flow[1].data_min_, series_train_philippe_nord_flow[1].data_max_))
#magnan
magnan_flow=data_split_flow_7loops(magnan)
## create daily time series train 
series_train_magnan_flow=daily_series(magnan_flow[0],170)
series_train_magnan_flow[0].shape
print('Min: %f, Max: %f' % (series_train_magnan_flow[1].data_min_, series_train_magnan_flow[1].data_max_))
#grinda
grinda_flow=data_split_flow_7loops(grinda)
## create daily time series train 
series_train_grinda_flow=daily_series(grinda_flow[0],170)
series_train_grinda_flow[0].shape
print('Min: %f, Max: %f' % (series_train_grinda_flow[1].data_min_, series_train_grinda_flow[1].data_max_))
#augustin
augustin_flow=data_split_flow_7loops(augustin)
## create daily time series train 
series_train_augustin_flow=daily_series(augustin_flow[0],170)
series_train_augustin_flow[0].shape
print('Min: %f, Max: %f' % (series_train_augustin_flow[1].data_min_, series_train_augustin_flow[1].data_max_))
#gloria
gloria_flow=data_split_flow_7loops(gloria)
## create daily time series train 
series_train_gloria_flow=daily_series(gloria_flow[0],170)
series_train_gloria_flow[0].shape
print('Min: %f, Max: %f' % (series_train_gloria_flow[1].data_min_, series_train_gloria_flow[1].data_max_))
#philippe sud 
philippe_sud_flow=data_split_flow_7loops(philippe_sud)
## create daily time series train 
series_train_philippe_sud_flow=daily_series(philippe_sud_flow[0],170)
series_train_philippe_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_sud_flow[1].data_min_, series_train_philippe_sud_flow[1].data_max_))
#cimiez_sud
cimiez_sud_flow=data_split_flow_7loops(cimiez_sud)
## create daily time series train 
series_train_cimiez_sud_flow=daily_series(cimiez_sud_flow[0],170)
series_train_cimiez_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_sud_flow[1].data_min_, series_train_cimiez_sud_flow[1].data_max_))
#multivariate time series train
multivariate=np.dstack((series_train_cimiez_nord_flow[0],series_train_philippe_nord_flow[0],series_train_magnan_flow[0],series_train_gloria_flow[0],series_train_philippe_sud_flow[0],series_train_cimiez_sud_flow[0],series_train_grinda_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#TEST
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
multivariate_test=np.dstack((series_test_cimiez_nord_flow[0],series_test_philippe_nord_flow[0],series_test_magnan_flow[0],series_test_gloria_flow[0],series_test_philippe_sud_flow[0],series_test_cimiez_sud_flow[0],series_test_grinda_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)

#CLUSTERING

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw




#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train,y=None)

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test)

#all
#train
# from 1/1 to 13/1 
first_mid=pd.date_range('1/1/2019', periods=13, freq='D')
#from 15/1 to 16/1
second_mid=pd.date_range('1/15/2019', periods=2, freq='D')
#from 18/1 to  26/1
third_mid=pd.date_range('1/18/2019', periods=9, freq='D')
#from 28/1 to  2/2
third_mid_bis=pd.date_range('1/28/2019', periods=6, freq='D')
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
# from 14/11 to 24/11
fifthteen_mid=pd.date_range('11/14/2019', periods=11, freq='D')
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
fifthteen_mid=pd.Series(data=fifthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
eighteen_mid_bis=pd.Series(data=eighteen_mid_bis)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,fifthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid,eighteen_mid_bis],ignore_index=True)
len(index_train)


#6 loops 
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
#from 18/10 to 22/10
twelveth_mid=pd.date_range('10/18/2019', periods=5, freq='D')
# from 25/10 to 2/11
thirdteen_mid=pd.date_range('10/25/2019', periods=9, freq='D')
# from 4/11 to 12/11
fourthteen_mid=pd.date_range('11/4/2019', periods=9, freq='D')
# from 14/11 to 24/11
fifthteen_mid=pd.date_range('11/14/2019', periods=11, freq='D')
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
fifthteen_mid=pd.Series(data=fifthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
eighteen_mid_bis=pd.Series(data=eighteen_mid_bis)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,third_mid_tris,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,fifthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid,eighteen_mid_bis],ignore_index=True)
len(index_train)

#7 loops 
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
#from 18/10 to 22/10
twelveth_mid=pd.date_range('10/18/2019', periods=5, freq='D')
# from 25/10 to 2/11
thirdteen_mid=pd.date_range('10/25/2019', periods=9, freq='D')
# from 4/11 to 12/11
fourthteen_mid=pd.date_range('11/4/2019', periods=9, freq='D')
# from 14/11 to 24/11
fifthteen_mid=pd.date_range('11/14/2019', periods=11, freq='D')
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
fifthteen_mid=pd.Series(data=fifthteen_mid)
sixtheen_mid=pd.Series(data=sixtheen_mid)
seventeen_mid=pd.Series(data=seventeen_mid)
eighteen_mid=pd.Series(data=eighteen_mid)
eighteen_mid_bis=pd.Series(data=eighteen_mid_bis)
index_train=pd.concat([first_mid,second_mid,third_mid,third_mid_bis,fourth_mid,fifth_mid,sixth_mid,septh_mid,eight_mid,ninth_mid,tenth_mid,eleventh_mid,twelveth_mid,thirdteen_mid,fourthteen_mid,fifthteen_mid,sixtheen_mid,seventeen_mid,eighteen_mid,eighteen_mid_bis],ignore_index=True)
len(index_train)


#plot the result 
new=[]
for i in range(0,310):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)
    
for i in range(0,340):
    if new[i] == 0.05:
        new[i] =4
        
for i in range(0,340):
    if new[i] == 1:
        new[i] =0.05

for i in range(0,340):
    if new[i] ==4:
        new[i] =1

#assign at every day the cluster
events_train = pd.Series(new,index=index_train)
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis 2019 (train):débit de circulation loops', linewidth=2.3,dropzero=True,vmin=0) 

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


events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis 2020 (test):débit de circulation loops', linewidth=2.3,dropzero=True,vmin=0) 

df=pd.read_excel(r'occupancy.xlsx')

df=df.set_index("index")
df

#take only observations from 6 to 22  54
df_2019=df.between_time('5:59', '22:59')
df_2019  

df_2019.to_excel(r'occupancy 2019 fake.xlsx')

gloria=pd.read_excel(r"fake.xlsx",sheet_name=0)

philippe_sud=pd.read_excel(r"fake.xlsx",sheet_name=1)

cimiez_sud=pd.read_excel(r"fake.xlsx",sheet_name=2)

cimiez_nord=pd.read_excel(r"fake.xlsx",sheet_name=3) 

philippe_nord=pd.read_excel(r"fake.xlsx",sheet_name=4)

magnan=pd.read_excel(r"fake.xlsx",sheet_name=5)


#TRAIN
#cimiez nord
cimiez_nord_flow=data_split_occupancy_6loops_fake(cimiez_nord)
## create daily time series train 
series_train_cimiez_nord_flow=daily_series(cimiez_nord_flow[0],170)
series_train_cimiez_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_nord_flow[1].data_min_, series_train_cimiez_nord_flow[1].data_max_))
#philippe nord
philippe_nord_flow=data_split_occupancy_6loops_fake(philippe_nord)
## create daily time series train 
series_train_philippe_nord_flow=daily_series(philippe_nord_flow[0],170)
series_train_philippe_nord_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_nord_flow[1].data_min_, series_train_philippe_nord_flow[1].data_max_))
#magnan
magnan_flow=data_split_occupancy_6loops_fake(magnan)
## create daily time series train 
series_train_magnan_flow=daily_series(magnan_flow[0],170)
series_train_magnan_flow[0].shape
print('Min: %f, Max: %f' % (series_train_magnan_flow[1].data_min_, series_train_magnan_flow[1].data_max_))
#gloria
gloria_flow=data_split_occupancy_6loops_fake(gloria)
## create daily time series train 
series_train_gloria_flow=daily_series(gloria_flow[0],170)
series_train_gloria_flow[0].shape
print('Min: %f, Max: %f' % (series_train_gloria_flow[1].data_min_, series_train_gloria_flow[1].data_max_))
#philippe sud 
philippe_sud_flow=data_split_occupancy_6loops_fake(philippe_sud)
## create daily time series train 
series_train_philippe_sud_flow=daily_series(philippe_sud_flow[0],170)
series_train_philippe_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_philippe_sud_flow[1].data_min_, series_train_philippe_sud_flow[1].data_max_))
#cimiez_sud
cimiez_sud_flow=data_split_occupancy_6loops_fake(cimiez_sud)
## create daily time series train 
series_train_cimiez_sud_flow=daily_series(cimiez_sud_flow[0],170)
series_train_cimiez_sud_flow[0].shape
print('Min: %f, Max: %f' % (series_train_cimiez_sud_flow[1].data_min_, series_train_cimiez_sud_flow[1].data_max_))
#multivariate time series train
multivariate=np.dstack((series_train_cimiez_nord_flow[0],series_train_philippe_nord_flow[0],series_train_magnan_flow[0],series_train_gloria_flow[0],series_train_philippe_sud_flow[0],series_train_cimiez_sud_flow[0]))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#TEST
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
multivariate_test=np.dstack((series_test_cimiez_nord_flow[0],series_test_philippe_nord_flow[0],series_test_magnan_flow[0],series_test_gloria_flow[0],series_test_philippe_sud_flow[0],series_test_cimiez_sud_flow[0]))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)


np.set_printoptions(threshold=np.inf)

np.argwhere(np.isnan(multivariate_time_series_train))


#estimate the gamma hyperparameter 
gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) 

#fit the model on train data 
km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=multivariate_time_series_train, n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)

#predict train 
prediction_train=km_dba.fit_predict(multivariate_time_series_train,y=None)

#prediction test 
prediction_test=km_dba.predict(multivariate_time_series_test)

new=[]
for i in range(0,365):
    if prediction_train[i] == 0:
        y=0.05
    elif prediction_train[i] !=0: 
        y=prediction_train[i]
    new.append(y)


index_prova=pd.date_range('1/1/2019', periods=365, freq='D')

events_train = pd.Series(new,index=index_prova)
primo=events_train[0:13]
secondo=events_train[14:16]
terzo=events_train[17:26]
quarto=events_train[27:44]
quinto=events_train[46:49]
sesto=events_train[50:98]
settimo=events_train[100:113]
ottavo=events_train[114:141]
nono=events_train[142:182]
decimo=events_train[184:236]
undici=events_train[237:269]
dodici=events_train[270:276]
tredici=events_train[277:289]
quattordici=events_train[290:295]
quindici=events_train[297:306]
sedici=events_train[307:316]
diciasette=events_train[317:329]
diciotto=events_train[330:342]
diciannove=events_train[343:351]
venti=events_train[352:353]
ventuno=events_train[354:366]

events_train=pd.concat([primo,secondo,terzo,quarto,quinto,sesto,settimo,ottavo,nono,decimo,undici,dodici,tredici,quattordici,quindici,sedici,diciasette,diciotto,diciannove,venti,ventuno])
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis 2019 (train):occupancy rate loops', linewidth=2.3,dropzero=True,vmin=0) 


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


events_test = pd.Series(new,index=index_test)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='Accent', suptitle='TrafficData Voie Mathis 2020 (test):occupancy rate loops', linewidth=2.3,dropzero=True,vmin=0) 








