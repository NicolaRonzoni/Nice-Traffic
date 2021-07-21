#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:20:16 2021

@author: nronzoni
"""

import pandas as pd 
import numpy as np

## extract speed #

#first semester---->remember to change the name of the lane#

df=pd.read_excel(r'MAGNAN.COMPTAGE.PVOIE.VR.xlsx',sheet_name=2)

#second semester#

df1=pd.read_excel(r'MAGNAN.COMPTAGE.PVOIE.VR.xlsx',sheet_name=3)


sem_1_2019=df.loc[(df['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_2R") | (df['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_VL") | (df['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_PL")] 

sem_2_2019=df1.loc[(df1['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_2R") | (df1['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_VL") | (df1['NAME'] =="MAGNAN.COMPTAGE.PVOIE.VR.CPT6_PL")]

#remove second from datetime

sem_1_2019['RECTS'] = sem_1_2019['RECTS'].dt.floor('Min')

sem_2_2019['RECTS'] = sem_2_2019['RECTS'].dt.floor('Min')
#delete useless columns 

sem_1_2019=sem_1_2019.drop(columns=['CHRONO','NAME','BATT'])

sem_2_2019=sem_2_2019.drop(columns=['CHRONO','NAME','BATT'])

sem_1_2019=sem_1_2019.set_index('RECTS')

sem_2_2019=sem_2_2019.set_index('RECTS')

sem_2_2019=sem_2_2019.groupby(level=0).sum()

sem_1_2019=sem_1_2019.groupby(level=0).sum()


#aggregate first and second semester 
flow2019=pd.concat([sem_1_2019, sem_2_2019])

flow2019= flow2019.resample('6min').mean()

len(flow2019)
#
VL=flow2019
#VR
VR=flow2019
#VM 
#VM=flow2019

columns=['VL','VR']

loop=pd.DataFrame(columns=columns,index=flow2019.index)
loop['VL']=VL.values
loop['VR']=VR.values
#loop['VM']=VM.values
loop=loop.sum(axis=1)
loop.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/magnan 2020 flow.xlsx')

#remove second from datetime

df2=pd.read_excel(r'PHILIPPE.COMPTAGE.PVOIE.SUD.VR.xlsx',sheet_name=1)


sem_1_2019=df2.loc[(df2['NAME'] =="PHILIPPE.COMPTAGE.PVOIE.SUD.VR.CPT6")] 


#remove second from datetime

sem_1_2019['RECTS'] = sem_1_2019['RECTS'].dt.floor('Min')


#delete useless columns 

sem_1_2019=sem_1_2019.drop(columns=['CHRONO','NAME','BATT'])



sem_1_2019=sem_1_2019.set_index('RECTS')





#aggregate first and second semester 
flow2019=sem_1_2019

flow2019= flow2019.resample('6min').mean()

len(flow2019)
#
VL=flow2019
#VR
VR=flow2019
#VM 
#VM=flow2019

columns=['VL','VR']

loop=pd.DataFrame(columns=columns,index=flow2019.index)
loop['VL']=VL.values
loop['VR']=VR.values
#loop['VM']=VM.values
loop=loop.sum(axis=1)
loop.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/philippe sud 2020 flow.xlsx')

####################################################################################

###########################################################################################################
# Keep observations in the range 6 to 22:54

df_flow=pd.read_excel(r'philippe sud 2019 flow.xlsx')
df_flow_1=pd.read_excel(r'philippe sud 2020 flow.xlsx')

#2019
flow_series=pd.Series(data=df_flow[0].values,index=pd.to_datetime(df_flow['RECTS']))
#2020
flow_series_1=pd.Series(data=df_flow_1[0].values,index=pd.to_datetime(df_flow_1['RECTS']))

flow_6min_2019=flow_series.between_time('5:59', '22:59')
flow_6min_2019=flow_6min_2019.replace(180,np.nan)
flow_6min_2019=flow_6min_2019.replace(108,np.nan)
flow_6min_2019=flow_6min_2019.replace(6,np.nan)
flow_6min_2019=flow_6min_2019.replace(90,np.nan)
flow_6min_2019=flow_6min_2019.replace(99,np.nan)
flow_6min_2019=flow_6min_2019.replace(to_replace=0, method='ffill',limit=1)
flow_6min_2019=flow_6min_2019.replace(135,np.nan)
flow_6min_2019=flow_6min_2019.replace(22,np.nan)



flow_6min_2020=flow_series_1.between_time('5:59', '22:59')
flow_6min_2020=flow_6min_2020.replace(180,np.nan)
flow_6min_2020=flow_6min_2020.replace(108,np.nan)
flow_6min_2020=flow_6min_2020.replace(6,np.nan)
flow_6min_2020=flow_6min_2020.replace(90,np.nan)
flow_6min_2020=flow_6min_2020.replace(99,np.nan)
flow_6min_2020=flow_6min_2020.replace(to_replace=0, method='ffill',limit=1)
flow_6min_2020=flow_6min_2020.replace(135,np.nan)
flow_6min_2020=flow_6min_2020.replace(22,np.nan)


len(flow_6min_2019)
len(flow_6min_2020)
#20/6-26/6
flow_1=flow_6min_2020[3230:3230+1190]
#24/2-1/3
flow_2=flow_6min_2020[9180:9180+1190]
#23/3-29/3
flow_3=flow_6min_2020[13940:13940+1190]
#20/7-26/7
flow_4=flow_6min_2020[34170:+34170+1190]

df=pd.read_excel(r'philippe sud 2019.xlsx')

df_1=pd.read_excel(r' 2020.xlsx')

speed_series=pd.Series(data=df['speed'].values,index=pd.to_datetime(df['index']))
speed_series_1=pd.Series(data=df_1['speed'].values,index=pd.to_datetime(df_1['index']))

occupancy_series=pd.Series(data=df['occupancy'].values,index=pd.to_datetime(df['index']))
occupancy_series_1=pd.Series(data=df_1['occupancy'].values,index=pd.to_datetime(df_1['index']))

#2019
speed_6min_2019=speed_series.between_time('5:59', '22:59')
speed_6min_2019[speed_6min_2019 < 2] = np.nan
speed_6min_2019=speed_6min_2019.replace(12,np.nan)
speed_6min_2019=speed_6min_2019.replace(13,np.nan)
speed_6min_2019=speed_6min_2019.replace(15,np.nan)
speed_6min_2019=speed_6min_2019.replace(14,np.nan)
speed_6min_2019=speed_6min_2019.replace(to_replace=0, method='ffill',limit=1)
speed_6min_2019=speed_6min_2019.replace(11,np.nan)
speed_6min_2019=speed_6min_2019.replace(2,np.nan)
speed_6min_2019=speed_6min_2019.replace(0,np.nan)


#2020
speed_6min_2020=speed_series_1.between_time('5:59', '22:59')
speed_6min_2020[speed_6min_2020 < 2] = np.nan
speed_6min_2020=speed_6min_2020.replace(12,np.nan)
speed_6min_2020=speed_6min_2020.replace(13,np.nan)
speed_6min_2020=speed_6min_2020.replace(15,np.nan)
speed_6min_2020=speed_6min_2020.replace(14,np.nan)
speed_6min_2020=speed_6min_2020.replace(to_replace=0, method='ffill',limit=1)
speed_6min_2020=speed_6min_2020.replace(1,np.nan)
speed_6min_2020=speed_6min_2020.replace(11,np.nan)
speed_6min_2020=speed_6min_2020.replace(0,np.nan)



len(speed_6min_2019)
len(speed_6min_2020)
#20/6-26/6
speed_1=speed_6min_2020[3230:3230+1190]
#24/2-1/3
speed_2=speed_6min_2020[9180:9180+1190]
#23/3-29/3
speed_3=speed_6min_2020[13940:13940+1190]
#20/7-26/7
speed_4=speed_6min_2020[34170:+34170+1190]


#2019
occupancy_6min_2019=occupancy_series.between_time('5:59', '22:59')
occupancy_6min_2019=occupancy_6min_2019.replace(12,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(13,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(14,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(15,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(to_replace=0, method='ffill',limit=1)
occupancy_6min_2019=occupancy_6min_2019.replace(0.5,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(11,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(2,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(2.5,np.nan)
occupancy_6min_2019=occupancy_6min_2019.replace(0,np.nan)

occupancy_6min_2019[occupancy_6min_2019 == 0].count()

#2020
occupancy_6min_2020=occupancy_series_1.between_time('5:59', '22:59')
occupancy_6min_2020=occupancy_6min_2020.replace(12,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(13,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(14,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(15,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(to_replace=0, method='ffill',limit=1)
occupancy_6min_2020=occupancy_6min_2020.replace(1,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(0.5,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(11,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(2,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(2.5,np.nan)
occupancy_6min_2020=occupancy_6min_2020.replace(0,np.nan)

occupancy_6min_2020[occupancy_6min_2020 == 0].count()
len(occupancy_6min_2019)
len(occupancy_6min_2020)

#20/6-26/6
occupancy_1=occupancy_6min_2020[3230:3230+1190]
#24/2-1/3
occupancy_2=occupancy_6min_2020[9180:9180+1190]
#23/3-29/3
occupancy_3=occupancy_6min_2020[13940:13940+1190]
#20/7-26/7
occupancy_4=occupancy_6min_2020[34170:+34170+1190]

flow=pd.concat([flow_6min_2019,flow_1,flow_2,flow_3,flow_4])

speed=pd.concat([speed_6min_2019,speed_1,speed_2,speed_3,speed_4])

occupancy=pd.concat([occupancy_6min_2019,occupancy_1,occupancy_2,occupancy_3,occupancy_4])

columns=['flow','speed','occupancy']

loop=pd.DataFrame(columns=columns,index=flow.index)
loop['flow']=flow.values
loop['speed']=speed.values
loop['occupancy']=occupancy.values

loop

loop= loop.fillna(-1)


loop['day'] = loop.index.day
loop['month'] =loop.index.month
loop['year'] = loop.index.year
loop['hour'] = loop.index.hour
loop['minute'] = loop.index.minute
loop.reset_index(drop=True, inplace=True)
loop = loop[['year', 'month', 'day','hour','minute', 'flow','occupancy','speed']]
loop
#####
#if flow=0 then occupancy=0
loop['occupancy']= np.where( (loop['flow']==-1),-1,loop.occupancy)
#if occupancy=0 then flow=0
loop['flow']= np.where( (loop['occupancy']==-1),-1,loop.flow)
#if flow=0 then speed=max
loop['speed']=pd.to_numeric(loop['speed'])
loop['speed']= np.where( (loop['flow']==-1),-1,loop.speed)

loop
#replace Nan value with -1
loop.to_excel('/Users/nronzoni/Desktop/Voie Mathis/loop detector/philippe sud.xlsx')

loop

#1
df_augustin=pd.read_excel(r'/Users/nronzoni/Desktop/Voie Mathis/loop detector/cimiez nord.xlsx')
#2
df_gloria=pd.read_excel(r'/Users/nronzoni/Desktop/Voie Mathis/loop detector/philippe nord.xlsx')
#3
df_philippesud=pd.read_excel(r'/Users/nronzoni/Desktop/Voie Mathis/loop detector/magnan.xlsx')
#4
df_cimiezsud=pd.read_excel(r'/Users/nronzoni/Desktop/Voie Mathis/loop detector/grinda.xlsx')



writer = pd.ExcelWriter('/Users/nronzoni/Desktop/Voie Mathis/loop detector/Voie Mathis North direction.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_augustin.to_excel(writer, sheet_name='Cimiez Nord')
df_gloria.to_excel(writer, sheet_name='Philippe Nord')
df_philippesud.to_excel(writer, sheet_name='Magnan')
df_cimiezsud.to_excel(writer, sheet_name='Grinda')


# Close the Pandas Excel writer and output the Excel file.
writer.save()

########################################################################################################################
#ramps
df=pd.read_csv(r'CIMIEZ.COMPTAGE.ENTREE.csv')
df

df['NAME']
#speed
speed_VL=df_VL.loc[(df_VL['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VL.VIT")] 

speed_VR=df_VR.loc[(df_VR['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VR.VIT")] 
#flow
flow_VL=df_VL.loc[(df_VL['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VL.CPT6_2R") | (df['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VL.CPT6_VL") | (df['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VL.CPT6_PL")]
flow_VR=df_VR.loc[(df_VR['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VR.CPT6_2R") | (df['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VR.CPT6_VL") | (df['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VR.CPT6_PL")]  
#occupancy rate 
tax_VL=df_VL.loc[(df_VL['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VL.TAUX")] 
tax_VR=df_VR.loc[(df_VR['NAME'] =="ELNOUZAH.COMPTAGE.PVOIE.VR.TAUX")] 


speed_VL['RECTS'] = pd.to_datetime(speed_VL['RECTS'], errors='coerce')
speed_VR['RECTS'] = pd.to_datetime(speed_VR['RECTS'], errors='coerce')

flow_VL['RECTS'] = pd.to_datetime(flow_VL['RECTS'], errors='coerce')
flow_VR['RECTS'] = pd.to_datetime(flow_VR['RECTS'], errors='coerce')

tax_VL['RECTS'] = pd.to_datetime(tax_VL['RECTS'], errors='coerce')
tax_VR['RECTS'] = pd.to_datetime(tax_VR['RECTS'], errors='coerce')

speed_VL['RECTS'] = speed_VL['RECTS'].dt.floor('Min')
speed_VR['RECTS'] = speed_VR['RECTS'].dt.floor('Min')

flow_VL['RECTS'] = flow_VL['RECTS'].dt.floor('Min')
flow_VR['RECTS'] = flow_VR['RECTS'].dt.floor('Min')

tax_VL['RECTS'] = tax_VL['RECTS'].dt.floor('Min')
tax_VR['RECTS'] = tax_VR['RECTS'].dt.floor('Min')


#delete useless columns 

speed_VL=speed_VL.drop(columns=['CHRONO','NAME','BATT'])
speed_VR=speed_VR.drop(columns=['CHRONO','NAME','BATT'])

flow_VL=flow_VL.drop(columns=['CHRONO','NAME','BATT'])
flow_VR=flow_VR.drop(columns=['CHRONO','NAME','BATT'])

tax_VL=tax_VL.drop(columns=['CHRONO','NAME','BATT'])
tax_VR=tax_VR.drop(columns=['CHRONO','NAME','BATT'])

#fill jumps in the series 
#delete duplicate for speed and taux 

speed_VL=speed_VL.drop_duplicates(subset='RECTS', keep='first', inplace=False, ignore_index=False)
speed_VR=speed_VR.drop_duplicates(subset='RECTS', keep='first', inplace=False, ignore_index=False)


tax_VL=tax_VL.drop_duplicates(subset='RECTS', keep='first', inplace=False, ignore_index=False)
tax_VR=tax_VR.drop_duplicates(subset='RECTS', keep='first', inplace=False, ignore_index=False)
#fill jumps in the series 
tax_VL=tax_VL.set_index('RECTS')
tax_VL=tax_VL.asfreq(freq='1Min', fill_value=np.nan)
len(tax_VL)
tax_VR=tax_VR.set_index('RECTS')
tax_VR=tax_VR.asfreq(freq='1Min', fill_value=np.nan)
len(tax_VR)

tax_VL=tax_VL.resample('6T').mean()
tax_VR=tax_VR.resample('6T').mean()


speed_VL=speed_VL.set_index('RECTS')
speed_VL=speed_VL.asfreq(freq='1Min', fill_value=np.nan)
len(speed_VL)

speed_VR=speed_VR.set_index('RECTS')
speed_VR=speed_VR.asfreq(freq='1Min', fill_value=np.nan)
len(speed_VR)

speed_VL=speed_VL.resample('6T').mean()
speed_VR=speed_VR.resample('6T').mean()



# flow 
flow_VL=flow_VL.set_index('RECTS')
flow_VR=flow_VR.set_index('RECTS')
#group all different vehicles type

flow_VL=flow_VL.groupby(level=0).sum()
flow_VR=flow_VR.groupby(level=0).sum()

#fill jumps in the series 

flow_VL= flow_VL.resample('6min').mean()
flow_VR= flow_VR.resample('6min').mean()

len(flow_VL)
len(flow_VR)

columns=['VL','VR']

flow=pd.DataFrame(columns=columns,index=flow_VR.index)
flow['VL']=flow_VL.values
flow['VR']=flow_VR.values
#loop['VM']=VM.values
flow=flow.sum(axis=1)
flow.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/ELNOUZAH flow.xlsx')

speed_VR=speed_VR[:-463]

speed=pd.DataFrame(columns=columns,index=speed_VR.index)
speed['VL']=speed_VL.values
speed['VR']=speed_VR.values
#loop['VM']=VM.values
speed=speed.sum(axis=1)
speed.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/ELNOUZAH speed.xlsx')

tax_VR=tax_VR[:-463]

tax=pd.DataFrame(columns=columns,index=tax_VR.index)
tax['VL']=tax_VL.values
tax['VR']=tax_VR.values
#loop['VM']=VM.values
tax=tax.sum(axis=1)
tax.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/ELNOUZAH occupancy.xlsx')


speed_series_6min=speed_series_6.between_time('4:59', '23:59')

occupancy_series_6min.plot()

occupancy_series_6min.plot(kind='box')

speed_series_6min.plot()

speed_series_6min.plot(kind='box')

# create the series for imputation and aggregation 

index=pd.date_range('2020-01-01',periods=527040, freq='1min')


speed_series_2019=pd.Series(data=filled["NVAL"].values,index=index)

speed_series_2019.plot()

speed_series_2019.plot(kind='box')


speed_series_2019.isnull().sum() * 100 / len(speed_series_2019)

#imputation 

speed_series_2019=speed_series_2019.ffill(axis=0,limit=20)

speed_series_2019.isnull().sum() * 100 / len(speed_series_2019)

# save series 
cleaned1=speed_series_2019.to_frame()
cleaned1=cleaned1.rename(columns={0:"speed VR"})
len(cleaned1)
cleaned1.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/philippe nord 2020 speed VR.xlsx')

sem_1_2019_occupancy=df.loc[df['NAME'] =="PHILIPPE.COMPTAGE.PVOIE.SUD.VR.TAUX"]


sem_2_2019_occupancy=df1.loc[df1['NAME'] =="PHILIPPE.COMPTAGE.PVOIE.NORD.VR.TAUX"]

#aggregate first and second semester 
occupancy2019=pd.concat([sem_1_2019_occupancy, sem_2_2019_occupancy])

#remove second from datetime

occupancy2019['RECTS'] = occupancy2019['RECTS'].dt.floor('Min')

#delete useless columns 

occupancy2019=occupancy2019.drop(columns=['CHRONO','NAME','BATT'])

#delete duplicate 

occupancy2019=occupancy2019.drop_duplicates(subset='RECTS', keep='first', inplace=False, ignore_index=False)

#ADD TIME and missing to fill jumps in the series 

filled2 = occupancy2019.set_index('RECTS')
filled2=filled2.asfreq(freq='1Min', fill_value=np.nan)
len(filled2)


index=pd.date_range('2020-01-01',periods=527040, freq='1min')

# create the series for imputation

occuppancy_series_2019=pd.Series(data=filled2["NVAL"].values,index=index)

occuppancy_series_2019.plot()

occuppancy_series_2019.plot(kind='box')


occuppancy_series_2019.isnull().sum() * 100 / len(occuppancy_series_2019)

#imputation 

occuppancy_series_2019=occuppancy_series_2019.ffill(axis=0,limit=20)

occuppancy_series_2019.isnull().sum() * 100 / len(occuppancy_series_2019)

# save series 
cleaned2=occuppancy_series_2019.to_frame()
cleaned2=cleaned2.rename(columns={0:"occupancy VR"})
len(cleaned2)
cleaned2.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/philippe sud 2020 occupancy VR.xlsx')



################################################################################
# import the data 
occupancy=pd.read_excel(r'cimiez sud 2020 occupancy.xlsx')

occupancy

speed=pd.read_excel(r'cimiez sud 2020 speed.xlsx')

speed

# put timestamp as index #

occupancy =occupancy.set_index('Unnamed: 0')

speed=speed.set_index('Unnamed: 0')

# mean of the lanes
occupancy=occupancy.mean(axis=1)

speed=speed.mean(axis=1)

index=pd.date_range('2020-01-01',periods=527040, freq='1min')

occupancy_series=pd.Series(data=occupancy.values,index=index)
speed_series=pd.Series(data=speed.values,index=index)

# 6 minute averages

occupancy_series_6=occupancy_series.resample('6T').mean()


speed_series_6=speed_series.resample('6T').mean()



### percentage of missing in the occupancy 
occupancy_series_6min.isnull().sum() * 100 / len(occupancy_series_6min)

### percentage of missing in the speed 
speed_series_6min.isnull().sum() * 100 / len(speed_series_6min)

ready=pd.concat([occupancy_series_6min,speed_series_6min], axis=1).reset_index()
len(ready)
ready1=ready.rename(columns={0:"occupancy", 1:"speed"})
len(ready1)
ready1.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/cimiez sud 2020.xlsx')



occupancy_series_6min.to_excel('/Users/nronzoni/Downloads/DataNice/Voie Mathis/DonnÇes 1-6min/first step/second step/philippe sud 2020.xlsx')



#remove outliers 
threshold=filled['NVAL'].quantile(0.75)+1.5*(filled['NVAL'].quantile(0.75)-filled['NVAL'].quantile(0.25))

filled['NVAL']= np.where( (filled['NVAL']>threshold), np.nan, filled.NVAL)




