#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:26:38 2021

@author: nronzoni
"""

import pandas as pd 
import numpy as np

###################################REMOVE DUPLICATE#######################
#load the data

df=pd.read_excel(r'occupacy C C094 2020cleaned.xlsx')


len(df)


#remove second from datetime if needed

remove=df

remove['timestamp'] = remove['timestamp'].dt.floor('Min')

remove

df=remove

#remove duplicate 
df_debit=df.drop_duplicates(subset='timestamp', keep='first', inplace=False, ignore_index=False)

len(df_debit)

#save it in a new xlsx file
df_debit.to_excel('/Users/nronzoni/Downloads/DataNice/Promenade des Anglais/2019/occupacy C077 C094 2019cleaned.xlsx', index = False)

#################################DELETE USELESS COLUMNS ##############################
# delete columns that do not correspond to any lane in the road

df=pd.read_excel(r'occupacy C077 C094 2019 cleaned.xlsx')
df.shape

df=df.drop(columns=['C077TO_3_VAL0','C094TO_3_VAL0'])

df.shape

df.to_excel('/Users/nronzoni/Downloads/DataNice/Promenade des Anglais/2019/cleaned/occupacy C077 C094 2019 cleaned.xlsx', index = False)

############################### ABERRANT DATA DETECTION ##################################
df=pd.read_excel(r'C599 2020cleaned.xlsx')
df.info()
# occupancy as missing if the occupancy rate is 100% and have a flow value  greater or equal to 0


df['C599TO_0_VAL0']= np.where( (df['C599TO_0_VAL0']>50 ), np.nan, df.C599TO_0_VAL0)

df['C599TO_1_VAL0']= np.where( (df['C599TO_1_VAL0']>50), np.nan, df.C599TO_1_VAL0)

df['C599TO_2_VAL0']= np.where( (df['C599TO_2_VAL0']>50), np.nan, df.C599TO_2_VAL0)

#df['C601TO_3_VAL0']= np.where((df['C601TO_3_VAL0']>50), np.nan, df.C601TO_3_VAL0)

# flow as missing if the flow is 0 and have a occupancy rate greater than 0 
df['C599DEBIT_0_VAL0']= np.where( (df['C599DEBIT_0_VAL0']>60) & (df['C599TO_0_VAL0'] ==100 ), np.nan, df.C599DEBIT_0_VAL0)

df['C599DEBIT_1_VAL0']= np.where( (df['C599DEBIT_1_VAL0']>60) & (df['C599TO_1_VAL0'] ==100 ), np.nan, df.C599DEBIT_1_VAL0)

df['C599DEBIT_2_VAL0']= np.where( (df['C599DEBIT_2_VAL0']>60) & (df['C599TO_2_VAL0'] ==100 ), np.nan, df.C599DEBIT_2_VAL0)

#df['C601DEBIT_3_VAL0']= np.where( (df['C601DEBIT_3_VAL0']>60) & (df['C601TO_3_VAL0'] ==100 ), np.nan, df.C601DEBIT_3_VAL0)                         

df.info()
##### observation from 6/2 only 2020 
df=df.iloc[1330:]
df
len(df)

###################### AGGREGATION OF THE DIFFERENT LANES 

df["debit"]=df["C599DEBIT_0_VAL0"]+df["C599DEBIT_1_VAL0"]+df["C599DEBIT_2_VAL0"]#+df["C613DEBIT_3_VAL0"]
##### occupacy 

df["occupacy"]=df["C599TO_0_VAL0"]+df["C599TO_1_VAL0"]+df["C599TO_2_VAL0"]#+df["C601TO_3_VAL0"]

df.shape

df= df.drop(['C599DEBIT_0_VAL0','C599DEBIT_1_VAL0','C599DEBIT_2_VAL0','C599TO_0_VAL0','C599TO_1_VAL0','C599TO_2_VAL0'], axis=1)

df.shape

df.info()
############### fundamental diagram to decide threshold ################
import matplotlib.pyplot as plt 
plt.plot(df["occupacy"],df["debit"],".",markersize=1.3)
plt.xlabel(xlabel='occupancy rate')
plt.ylabel(ylabel='flow')
plt.title(label='fundamental diagram')

df['occupacy']= np.where( (df['occupacy']>100), np.nan, df.occupacy)

df['debit']= np.where( (df['occupacy']>100), np.nan, df.debit)

df['occupacy']= np.where( (df['debit']>4000), np.nan, df.occupacy)

df['debit']= np.where( (df['debit']>4000), np.nan, df.debit)

len(df)
######### ADD TIME and missing to fill jumps in the series 
filled = df.set_index('timestamp')
filled=filled.asfreq(freq='1Min', fill_value=np.nan)
len(filled)

########### time series for flow and occupancy to have 6-minute averages
index=pd.date_range('2020-02-06',periods=473761, freq='1min')


flow_series=pd.Series(data=filled["debit"].values,index=index)

occupancy_series=pd.Series(data=filled['occupacy'].values,index=index)

############################# 6minute averages ################################

series_flow_6=flow_series.resample('6T').mean()


series_occupancy_6=occupancy_series.resample('6T').mean()

### percentage of missing in the flow 
series_flow_6.isnull().sum() * 100 / len(series_flow_6)

### percentage of missing in the occupancy 
series_occupancy_6.isnull().sum() * 100 / len(series_occupancy_6)

################ look at where missing values are ##########################################


series_flow_6.plot(title='flow 2019 C009 detector')

series_occupancy_6.plot(title='occupancy 2019 C009 detector')

series_occupancy_6.plot(kind='box',title='occupacy')

series_flow_6.plot(kind='box',title='flow')


#############imputation of missing values with respect to the the temporal aligment

s1=series_flow_6.ffill(axis=0,limit=1)

s1=s1.interpolate(method='time',axis=0,limit=20)

s2=series_occupancy_6.ffill(axis=0,limit=1)

s2=s2.interpolate(method='time',axis=0,limit=20)


### percentage of missing in the flow 
s1.isnull().sum() * 100 / len(series_flow_6)

### percentage of missing in the occupancy 
s2.isnull().sum() * 100 / len(series_occupancy_6)
s1.isnull().sum() 

cleaned=pd.concat([s1, s2], axis=1).reset_index()
len(cleaned)
cleaned1=cleaned.rename(columns={0:"flow", 1:"occupancy"})
len(cleaned1)
cleaned1.to_excel('/Users/nronzoni/Downloads/DataNice/Promenade des Anglais/2020/C599 6min 2020.xlsx')



####### first lockdown impact 
# START 02/03 29/03
start_firstlockdown=series_debit[6000:12720]
start_firstlockdown.plot()

# END 15/05 5/06
end_firstlockdown=series_debit[23760:28800]
end_firstlockdown.plot()

####### second lockdown impact 

# 23/10 4/12
start_secondlockdown=debit[62400:72480]
start_secondlockdown.plot()

########### moving average #############
##### import data 
df= pd.read_excel(r"C601 6min 2019.xlsx")

df20= pd.read_excel("/Users/nronzoni/Downloads/DataNice/Promenade des Anglais/2020/C009 6min 20201.xlsx")

df20['SMA'] = df20['flow'].rolling(window=1680).mean()
df['EMA'] = df['flow'].ewm(adjust=False,span=1680).mean()

simply_moving_average=pd.Series(data=df20['SMA'].values, index=df20['index'])

exp_moving_average=pd.Series(data=df['EMA'].values, index=df['index'])

simply_moving_average.plot(title='SMA of 1 week C009 detector')

exp_moving_average.plot(title='EMA of 1 week C601 detector')

######### summer vs winter
# April 6/4 to 10/4 2020
ap20=df20['flow'][14400:15600]
len(ap20)

#may 18/5 to 22/5 2020
may20=df20['flow'][:]
len(may20)

#10/6 to 14/6 June  2019
june=df['flow'][2160:3360]
len(june)
#8/6 to 12/6 2020 
june20=df20['flow'][29520:30720]
len(june20)

# 8/7 to 12/7
july=df['occupancy'][8880:10080]
len(july)
  
 # 5/8 to 9/8
august=df['occupancy'][15600:16800]
len(august)
 # 3/8 to 7/8 2020 
august20=df20['flow'][42960:44160]
len(august20)
# 9/9 to 13/9 2019
september=df['flow'][24000:25200]
len(september)

# 7/9 to 11/9 2020
september20=df20['flow'][51360:52560]
len(september20)

# 18/11 to 22/11
november=df['flow'][40800:42000]
len(november)
# 9/12 to 13/12
december=df['occupancy'][44160:45360]
len(december)

mylist=list(range(0,1200))
import matplotlib.pyplot as plt
plt.plot(mylist,august,'#ff3399',label='from 5/8 to 9/8', linewidth=0.5,alpha=0.5)
plt.plot(mylist,july,'#cc33ff',label='from 8/7 to 12/7', linewidth=0.5)
plt.title('occupancy rate of C094 detector on different months')
ax=plt.gca()
ax.axes.xaxis.set_ticks([])
plt.xlabel('Mon             Tue                 Wed              Thur              Fri')
plt.legend(loc='lower left')
plt.show()

plt.plot(mylist,august20,'c-',label='from 3/8 to 7/8', linewidth=0.5)
plt.plot(mylist,september20,'#ff3399',label='from 7/9 to 11/9', linewidth=0.5)
plt.title('number of veh of C601 detector on different months')
ax=plt.gca()
ax.axes.xaxis.set_ticks([])
plt.xlabel('Mon             Tue                 Wed              Thur              Fri')
plt.legend(loc='lower left')
plt.show()




plt.plot(mylist,september,'r-',label='2019', linewidth=0.5)
plt.plot(mylist,september20,'c--',label='2020', linewidth=0.5)
plt.title('number of veh of C601 detector 2$^{nd}$ week of September')
ax=plt.gca()
ax.axes.xaxis.set_ticks([])
plt.xlabel('Mon             Tue                 Wed              Thur              Fri')
plt.legend(loc='lower left')
plt.show()

plt.plot(mylist,june,'r-',label='2019', linewidth=0.5)
plt.plot(mylist,june20,'c--',label='2020', linewidth=0.5)
plt.title('number of veh of C601 detector 2$^{nd}$ week of June')
ax=plt.gca()
ax.axes.xaxis.set_ticks([])
plt.xlabel('Mon             Tue                 Wed              Thur              Fri')
plt.legend(loc='lower left')
plt.show()













