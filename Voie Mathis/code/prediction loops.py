#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:43:30 2021

@author: nronzoni
"""
#################################flow 
multivariate_time_series_train.shape
first_day=classification_pred_same(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,10,20)
#prediction
first_day[2].shape
#ground_truth
first_day[3].shape
columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth','S57 flow (veh/h)','S57 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth','S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_34= pd.DataFrame(index=index.time, columns=columns)
df_34

Y_pred_S54=series_train_S54_flow[1].inverse_transform(first_day[2][:,:,0])
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day[3][:,0:1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_flow[1].inverse_transform(first_day[2][:,:,1])
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day[3][:,1:2])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_train_S56_flow[1].inverse_transform(first_day[2][:,:,2])
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day[3][:,2:3])
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_train_S57_flow[1].inverse_transform(first_day[2][:,:,3])
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day[3][:,3:4])
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_flow[1].inverse_transform(first_day[2][:,:,4])
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day[3][:,4:5])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_flow[1].inverse_transform(first_day[2][:,:,5])
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day[3][:,5:6])
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_train_S60_flow[1].inverse_transform(first_day[2][:,:,6])
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day[3][:,6:7])
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_flow[1].inverse_transform(first_day[2][:,:,7])
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day[3][:,7:8])
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])


df_34['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_34['S54 flow (veh/h) ground truth']=Y_test_S54
df_34['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_34['S1706 flow (veh/h) ground truth']=Y_test_S1706
df_34['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_34['S56 flow (veh/h) ground truth']=Y_test_S56
df_34['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_34['S57 flow (veh/h) ground truth']=Y_test_S57
df_34['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_34['S1707 flow (veh/h) ground truth']=Y_test_S1707
df_34['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_34['S59 flow (veh/h) ground truth']=Y_test_S59
df_34['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_34['S60 flow (veh/h) ground truth']=Y_test_S60
df_34['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_34['S61 flow (veh/h) ground truth']=Y_test_S61
df_34



df_30

df_31

df_32

df_33

df_34



# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /Classification_prediction_flow.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_30.to_excel(writer, sheet_name='10-2-2014 morning')
df_31.to_excel(writer, sheet_name='12-2-2014 morning')
df_32.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_33.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_34.to_excel(writer, sheet_name='10-9-2014 afternoon')

# Close the Pandas Excel writer and output the Excel file.
writer.save()



first_day_S54=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,0)
first_day_S1706=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,1)
first_day_S56=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,2)
first_day_S57=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,3)
first_day_S1707=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,4)
first_day_S59=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,5)
first_day_S60=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,6)
first_day_S61=SVR_pred_d(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],30,5,10,7)


Y_pred_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[1])
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[2])
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[1])
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[2])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[1])
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[2])
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[1])
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[2])
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[1])
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[2])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[1])
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[2])
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[1])
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[2])
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[1])
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[2])
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])


columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_35 = pd.DataFrame(index=index.time, columns=columns)
df_35
df_35['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_35['S54 flow (veh/h) ground truth']=Y_test_S54.reshape(-1,1)
df_35['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_35['S1706 flow (veh/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_35['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_35['S56 flow (veh/h) ground truth']=Y_test_S56.reshape(-1,1)
df_35['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_35['S57 flow (veh/h) ground truth']=Y_test_S57.reshape(-1,1)
df_35['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_35['S1707 flow (veh/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_35['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_35['S59 flow (veh/h) ground truth']=Y_test_S59.reshape(-1,1)
df_35['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_35['S60 flow (veh/h) ground truth']=Y_test_S60.reshape(-1,1)
df_35['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_35['S61 flow (veh/h) ground truth']=Y_test_S61.reshape(-1,1)
df_35

#10/2
df_31
#12/2
df_32
#22/3
df_33
#15/08
df_34
#10/09
df_35
#

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_flow.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_31.to_excel(writer, sheet_name='10-2-2014 morning')
df_32.to_excel(writer, sheet_name='12-2-2014 morning')
df_33.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_34.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_35.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

#################################################### speed ##########################
first_day=classification_pred_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[12:13,:,:],115,10,20)

#prediction
first_day[1].shape
#ground_truth
first_day[2].shape
columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth', 'S57 speed (km/h)','S57 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("14:30", periods=10, freq="6min")
df_39= pd.DataFrame(index=index.time, columns=columns)
df_39

Y_pred_S54=series_train_S54_speed[1].inverse_transform(first_day[1][:,:,0])
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day[2][:,0:1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54,Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_train_S1706_speed[1].inverse_transform(first_day[1][:,:,1])
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day[2][:,1:2])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706,Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_train_S56_speed[1].inverse_transform(first_day[1][:,:,2])
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day[2][:,2:3])
error_S56=math.sqrt(mean_squared_error(Y_test_S56,Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_train_S57_speed[1].inverse_transform(first_day[1][:,:,3])
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day[2][:,3:4])
error_S57=math.sqrt(mean_squared_error(Y_test_S57,Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_train_S1707_speed[1].inverse_transform(first_day[1][:,:,4])
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day[2][:,4:5])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707,Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_train_S59_speed[1].inverse_transform(first_day[1][:,:,5])
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day[2][:,5:6])
error_S59=math.sqrt(mean_squared_error(Y_test_S59,Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_train_S60_speed[1].inverse_transform(first_day[1][:,:,6])
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day[2][:,6:7])
error_S60=math.sqrt(mean_squared_error(Y_test_S60,Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_train_S61_speed[1].inverse_transform(first_day[1][:,:,7])
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day[2][:,7:8])
error_S61=math.sqrt(mean_squared_error(Y_test_S61,Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])




df_39['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_39['S54 speed (km/h) ground truth']=Y_test_S54
df_39['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_39['S1706 speed (km/h) ground truth']=Y_test_S1706
df_39['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_39['S56 speed (km/h) ground truth']=Y_test_S56
df_39['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_39['S57 speed (km/h) ground truth']=Y_test_S57
df_39['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_39['S1707 speed (km/h) ground truth']=Y_test_S1707
df_39['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_39['S59 speed (km/h) ground truth']=Y_test_S59
df_39['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_39['S60 speed (km/h) ground truth']=Y_test_S60
df_39['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_39['S61 speed (km/h) ground truth']=Y_test_S61
df_39



#10/2
df_36
#10/9
df_40
#12/2
df_37
#22/03
df_38
#15/08
df_39





# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /Classification_prediction_speed.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_36.to_excel(writer, sheet_name='10-2-2014 morning')
df_37.to_excel(writer, sheet_name='12-2-2014 morning')
df_38.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_39.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_40.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


################################ svr ################### 
first_day_S54=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,0)
first_day_S1706=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,1)
first_day_S56=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,2)
first_day_S57=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,3)
first_day_S1707=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,4)
first_day_S59=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,5)
first_day_S60=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,6)
first_day_S61=SVR_pred_d_speed(multivariate_time_series_train_speed_subset,multivariate_time_series_test_speed[18:19,:,:],95,5,10,7)


Y_pred_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[0])
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[1])
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[0])
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[1])
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[0])
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[1])
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[0])
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[1])
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[0])
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[1])
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[0])
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[1])
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[0])
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[1])
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[0])
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[1])
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])



columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth', 'S56 speed (km/h)','S56 speed (km/h) ground truth','S57 speed (km/h)','S57 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth','S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("14:30", periods=10, freq="6min")
df_44= pd.DataFrame(index=index.time, columns=columns)
df_44
df_44['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_44['S54 speed (km/h) ground truth']=Y_test_S54.reshape(-1,1)
df_44['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_44['S1706 speed (km/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_44['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_44['S56 speed (km/h) ground truth']=Y_test_S56.reshape(-1,1)
df_44['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_44['S57 speed (km/h) ground truth']=Y_test_S57.reshape(-1,1)
df_44['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_44['S1707 speed (km/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_44['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_44['S59 speed (km/h) ground truth']=Y_test_S59.reshape(-1,1)
df_44['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_44['S60 speed (km/h) ground truth']=Y_test_S60.reshape(-1,1)
df_44['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_44['S61 speed (km/h) ground truth']=Y_test_S61.reshape(-1,1)
df_44

#10/2
df_41
#10/9
df_45
#12/2
df_42
#22/3
df_43
#15/8
df_44

writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_speed.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_41.to_excel(writer, sheet_name='10-2-2014 morning')
df_42.to_excel(writer, sheet_name='12-2-2014 morning')
df_43.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_44.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_45.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


###############walk forward validation 

##################### FLOW 

first_day_S54=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,0)
first_day_S1706=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,1)
first_day_S56=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,2)
first_day_S57=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,3)
first_day_S1707=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,4)
first_day_S59=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,5)
first_day_S60=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,6)
first_day_S61=walk_forward_validation(multivariate_time_series_train,multivariate_time_series_test[23:24,:,:],5,30,7)


Y_pred_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[0].reshape(-1,1))
Y_test_S54=series_test_S54_flow[1].inverse_transform(first_day_S54[1].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[0].reshape(-1,1))
Y_test_S1706=series_test_S1706_flow[1].inverse_transform(first_day_S1706[1].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[0].reshape(-1,1))
Y_test_S56=series_test_S56_flow[1].inverse_transform(first_day_S56[1].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[0].reshape(-1,1))
Y_test_S57=series_test_S57_flow[1].inverse_transform(first_day_S57[1].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[0].reshape(-1,1))
Y_test_S1707=series_test_S1707_flow[1].inverse_transform(first_day_S1707[1].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[0].reshape(-1,1))
Y_test_S59=series_test_S59_flow[1].inverse_transform(first_day_S59[1].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[0].reshape(-1,1))
Y_test_S60=series_test_S60_flow[1].inverse_transform(first_day_S60[1].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[0].reshape(-1,1))
Y_test_S61=series_test_S61_flow[1].inverse_transform(first_day_S61[1].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])


columns = ['S54 flow (veh/h)','S54 flow (veh/h) ground truth','S1706 flow (veh/h)','S1706 flow (veh/h) ground truth','S56 flow (veh/h)','S56 flow (veh/h) ground truth', 'S57 flow (veh/h)','S57 flow (veh/h) ground truth','S1707 flow (veh/h)','S1707 flow (veh/h) ground truth', 'S59 flow (veh/h)','S59 flow (veh/h) ground truth', 'S60 flow (veh/h)','S60 flow (veh/h) ground truth','S61 flow (veh/h)','S61 flow (veh/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_50= pd.DataFrame(index=index.time, columns=columns)
df_50
df_50['S54 flow (veh/h)']=Y_pred_S54.reshape(-1,1)
df_50['S54 flow (veh/h) ground truth']=Y_test_S54.reshape(-1,1)
df_50['S1706 flow (veh/h)']=Y_pred_S1706.reshape(-1,1)
df_50['S1706 flow (veh/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_50['S56 flow (veh/h)']=Y_pred_S56.reshape(-1,1)
df_50['S56 flow (veh/h) ground truth']=Y_test_S56.reshape(-1,1)
df_50['S57 flow (veh/h)']=Y_pred_S57.reshape(-1,1)
df_50['S57 flow (veh/h) ground truth']=Y_test_S57.reshape(-1,1)
df_50['S1707 flow (veh/h)']=Y_pred_S1707.reshape(-1,1)
df_50['S1707 flow (veh/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_50['S59 flow (veh/h)']=Y_pred_S59.reshape(-1,1)
df_50['S59 flow (veh/h) ground truth']=Y_test_S59.reshape(-1,1)
df_50['S60 flow (veh/h)']=Y_pred_S60.reshape(-1,1)
df_50['S60 flow (veh/h) ground truth']=Y_test_S60.reshape(-1,1)
df_50['S61 flow (veh/h)']=Y_pred_S61.reshape(-1,1)
df_50['S61 flow (veh/h) ground truth']=Y_test_S61.reshape(-1,1)
df_50



#10/2
df_46
#12/2
df_47
#22/3
df_48
#15/08
df_49
#10/09
df_50
#

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /WALKFORWARD_SupportVectorRegression_prediction_flow.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_46.to_excel(writer, sheet_name='10-2-2014 morning')
df_47.to_excel(writer, sheet_name='12-2-2014 morning')
df_48.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_49.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_50.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

##################### speed


first_day_S54=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,0)
first_day_S1706=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,1)
first_day_S56=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,2)
first_day_S57=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,3)
first_day_S1707=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,4)
first_day_S59=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,5)
first_day_S60=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,6)
first_day_S61=walk_forward_validation(multivariate_time_series_train_speed,multivariate_time_series_test_speed[2:3,:,:],5,30,7)


Y_pred_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[0].reshape(-1,1))
Y_test_S54=series_test_S54_speed[1].inverse_transform(first_day_S54[1].reshape(-1,1))
error_S54=math.sqrt(mean_squared_error(Y_test_S54.reshape(-1,1),Y_pred_S54.reshape(-1,1)))
Y_pred_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[0].reshape(-1,1))
Y_test_S1706=series_test_S1706_speed[1].inverse_transform(first_day_S1706[1].reshape(-1,1))
error_S1706=math.sqrt(mean_squared_error(Y_test_S1706.reshape(-1,1),Y_pred_S1706.reshape(-1,1)))
Y_pred_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[0].reshape(-1,1))
Y_test_S56=series_test_S56_speed[1].inverse_transform(first_day_S56[1].reshape(-1,1))
error_S56=math.sqrt(mean_squared_error(Y_test_S56.reshape(-1,1),Y_pred_S56.reshape(-1,1)))
Y_pred_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[0].reshape(-1,1))
Y_test_S57=series_test_S57_speed[1].inverse_transform(first_day_S57[1].reshape(-1,1))
error_S57=math.sqrt(mean_squared_error(Y_test_S57.reshape(-1,1),Y_pred_S57.reshape(-1,1)))
Y_pred_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[0].reshape(-1,1))
Y_test_S1707=series_test_S1707_speed[1].inverse_transform(first_day_S1707[1].reshape(-1,1))
error_S1707=math.sqrt(mean_squared_error(Y_test_S1707.reshape(-1,1),Y_pred_S1707.reshape(-1,1)))
Y_pred_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[0].reshape(-1,1))
Y_test_S59=series_test_S59_speed[1].inverse_transform(first_day_S59[1].reshape(-1,1))
error_S59=math.sqrt(mean_squared_error(Y_test_S59.reshape(-1,1),Y_pred_S59.reshape(-1,1)))
Y_pred_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[0].reshape(-1,1))
Y_test_S60=series_test_S60_speed[1].inverse_transform(first_day_S60[1].reshape(-1,1))
error_S60=math.sqrt(mean_squared_error(Y_test_S60.reshape(-1,1),Y_pred_S60.reshape(-1,1)))
Y_pred_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[0].reshape(-1,1))
Y_test_S61=series_test_S61_speed[1].inverse_transform(first_day_S61[1].reshape(-1,1))
error_S61=math.sqrt(mean_squared_error(Y_test_S61.reshape(-1,1),Y_pred_S61.reshape(-1,1)))

error=mean([error_S54,error_S1706,error_S56,error_S57,error_S1707,error_S59,error_S60,error_S61])


columns = ['S54 speed (km/h)','S54 speed (km/h) ground truth','S1706 speed (km/h)','S1706 speed (km/h) ground truth','S56 speed (km/h)','S56 speed (km/h) ground truth','S57 speed (km/h)','S57 speed (km/h) ground truth','S1707 speed (km/h)','S1707 speed (km/h) ground truth', 'S59 speed (km/h)','S59 speed (km/h) ground truth', 'S60 speed (km/h)','S60 speed (km/h) ground truth','S61 speed (km/h)','S61 speed (km/h) ground truth']
index=pd.date_range("08:00", periods=10, freq="6min")
df_52= pd.DataFrame(index=index.time, columns=columns)
df_52
df_52['S54 speed (km/h)']=Y_pred_S54.reshape(-1,1)
df_52['S54 speed (km/h) ground truth']=Y_test_S54.reshape(-1,1)
df_52['S1706 speed (km/h)']=Y_pred_S1706.reshape(-1,1)
df_52['S1706 speed (km/h) ground truth']=Y_test_S1706.reshape(-1,1)
df_52['S56 speed (km/h)']=Y_pred_S56.reshape(-1,1)
df_52['S56 speed (km/h) ground truth']=Y_test_S56.reshape(-1,1)
df_52['S57 speed (km/h)']=Y_pred_S57.reshape(-1,1)
df_52['S57 speed (km/h) ground truth']=Y_test_S57.reshape(-1,1)
df_52['S1707 speed (km/h)']=Y_pred_S1707.reshape(-1,1)
df_52['S1707 speed (km/h) ground truth']=Y_test_S1707.reshape(-1,1)
df_52['S59 speed (km/h)']=Y_pred_S59.reshape(-1,1)
df_52['S59 speed (km/h) ground truth']=Y_test_S59.reshape(-1,1)
df_52['S60 speed (km/h)']=Y_pred_S60.reshape(-1,1)
df_52['S60 speed (km/h) ground truth']=Y_test_S60.reshape(-1,1)
df_52['S61 speed (km/h)']=Y_pred_S61.reshape(-1,1)
df_52['S61 speed (km/h) ground truth']=Y_test_S61.reshape(-1,1)
df_52

#10/2
df_51
#12/2
df_52
#22/3
df_53
#15/8
df_54
#10/9
df_55


writer = pd.ExcelWriter('/Users/nronzoni/Desktop/TrafficData Minnesota/WALKFORWARD_SupportVectorRegression_prediction_speed.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_51.to_excel(writer, sheet_name='10-2-2014 morning')
df_52.to_excel(writer, sheet_name='12-2-2014 morning')
df_53.to_excel(writer, sheet_name='22-3-2014 afternoon')
df_54.to_excel(writer, sheet_name='15-8-2014 afternoon')
df_55.to_excel(writer, sheet_name='10-9-2014 morning')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

#########PLOTS

#fix the range of the prediction 
x=index_third_period=pd.date_range('2014-02-10 08:00:00',periods=10, freq='6min')
len(x)
x=x.strftime("%H:%M")

#take the day you would like to plot 
df_SVR = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_flow.xlsx', sheet_name='10-9-2014 morning')

df_CLASS = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /Classification_prediction_flow.xlsx', sheet_name='10-9-2014 morning') 

#fix upper bound and lower bound for the flow 
df_CLASS.min(axis=0)
df_SVR.min(axis=0)
df_CLASS.max(axis=0)
df_SVR.max(axis=0)

minimum=4500
maximum=8500
#ground truth svr classifaction prediction 
S54_SVR=df_SVR['S54 flow (veh/h)'].values
S54_CLASS=df_CLASS['S54 flow (veh/h)'].values
S54_GT=df_SVR['S54 flow (veh/h) ground truth'].values
S1706_SVR=df_SVR['S1706 flow (veh/h)'].values
S1706_CLASS=df_CLASS['S1706 flow (veh/h)'].values
S1706_GT=df_SVR['S1706 flow (veh/h) ground truth'].values
S56_SVR=df_SVR['S56 flow (veh/h)'].values
S56_CLASS=df_CLASS['S56 flow (veh/h)'].values
S56_GT=df_SVR['S56 flow (veh/h) ground truth'].values
S57_SVR=df_SVR['S57 flow (veh/h)'].values
S57_CLASS=df_CLASS['S57 flow (veh/h)'].values
S57_GT=df_SVR['S57 flow (veh/h) ground truth'].values
S1707_SVR=df_SVR['S1707 flow (veh/h)'].values
S1707_CLASS=df_CLASS['S1707 flow (veh/h)'].values
S1707_GT=df_SVR['S1707 flow (veh/h) ground truth'].values
S59_SVR=df_SVR['S59 flow (veh/h)'].values
S59_CLASS=df_CLASS['S59 flow (veh/h)'].values
S59_GT=df_SVR['S59 flow (veh/h) ground truth'].values
S60_SVR=df_SVR['S60 flow (veh/h)'].values
S60_CLASS=df_CLASS['S60 flow (veh/h)'].values
S60_GT=df_SVR['S60 flow (veh/h) ground truth'].values
S61_SVR=df_SVR['S61 flow (veh/h)'].values
S61_CLASS=df_CLASS['S61 flow (veh/h)'].values
S61_GT=df_SVR['S61 flow (veh/h) ground truth'].values
GT=np.concatenate((S54_GT,S1706_GT,S56_GT,S57_GT,S1707_GT,S59_GT,S60_GT, S61_GT), axis=None)
SVR=np.concatenate((S54_SVR,S1706_SVR,S56_SVR,S57_SVR,S1707_SVR,S59_SVR,S60_SVR, S61_SVR), axis=None)
CLASS=np.concatenate((S54_CLASS,S1706_CLASS,S56_CLASS,S57_CLASS,S1707_CLASS,S59_CLASS,S60_CLASS, S61_CLASS), axis=None)
len(GT)
error_SVR=math.sqrt(mean_squared_error(GT,SVR))
round(error_SVR,4)
error_CLASS=math.sqrt(mean_squared_error(GT,CLASS))
round(error_CLASS,4)


plt.figure(figsize=(35,25))
plt.subplot(2,4,1)
plt.plot(x,S54_SVR,'r-',label='SVR prediction')
plt.plot(x,S54_CLASS,'b-',label='Classification prediction')
plt.plot(x,S54_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S54')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,2)
plt.plot(x,S1706_SVR,'r-',label='SVR prediction')
plt.plot(x,S1706_CLASS,'b-',label='Classification prediction')
plt.plot(x,S1706_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1706')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,3)
plt.plot(x,S56_SVR,'r-',label='SVR prediction')
plt.plot(x,S56_CLASS,'b-',label='Classification prediction')
plt.plot(x,S56_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S56')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,4)
plt.plot(x,S57_SVR,'r-',label='SVR prediction')
plt.plot(x,S57_CLASS,'b-',label='Classification prediction')
plt.plot(x,S57_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S57')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,5)
plt.plot(x,S1707_SVR,'r-',label='SVR prediction')
plt.plot(x,S1707_CLASS,'b-',label='Classification prediction')
plt.plot(x,S1707_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1707')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,6)
plt.plot(x,S59_SVR,'r-',label='SVR prediction')
plt.plot(x,S59_CLASS,'b-',label='Classification prediction')
plt.plot(x,S59_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S59')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,7)
plt.plot(x,S60_SVR,'r-',label='SVR prediction')
plt.plot(x,S60_CLASS,'b-',label='Classification prediction')
plt.plot(x,S60_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S60')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,8)
plt.plot(x,S61_SVR,'r-',label='SVR prediction')
plt.plot(x,S61_CLASS,'b-',label='Classification prediction')
plt.plot(x,S61_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S61')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.suptitle("12/2/2014 flow predictions: loops",fontsize=24, y=0.93)
plt.show()



df_SVR = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_speed.xlsx', sheet_name='10-9-2014 morning')

df_CLASS = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /Classification_prediction_speed.xlsx', sheet_name='10-9-2014 morning') 

#fix upper bound and lower bound for the flow 
df_CLASS.min(axis=0)
df_SVR.min(axis=0)
df_CLASS.max(axis=0)
df_SVR.max(axis=0)

minimum=10
maximum=125
#ground truth svr classifaction prediction 
S54_SVR=df_SVR['S54 speed (km/h)'].values
S54_CLASS=df_CLASS['S54 speed (km/h)'].values
S54_GT=df_SVR['S54 speed (km/h) ground truth'].values
S1706_SVR=df_SVR['S1706 speed (km/h)'].values
S1706_CLASS=df_CLASS['S1706 speed (km/h)'].values
S1706_GT=df_SVR['S1706 speed (km/h) ground truth'].values
S56_SVR=df_SVR['S56 speed (km/h)'].values
S56_CLASS=df_CLASS['S56 speed (km/h)'].values
S56_GT=df_SVR['S56 speed (km/h) ground truth'].values
S57_SVR=df_SVR['S57 speed (km/h)'].values
S57_CLASS=df_CLASS['S57 speed (km/h)'].values
S57_GT=df_SVR['S57 speed (km/h) ground truth'].values
S1707_SVR=df_SVR['S1707 speed (km/h)'].values
S1707_CLASS=df_CLASS['S1707 speed (km/h)'].values
S1707_GT=df_SVR['S1707 speed (km/h) ground truth'].values
S59_SVR=df_SVR['S59 speed (km/h)'].values
S59_CLASS=df_CLASS['S59 speed (km/h)'].values
S59_GT=df_SVR['S59 speed (km/h) ground truth'].values
S60_SVR=df_SVR['S60 speed (km/h)'].values
S60_CLASS=df_CLASS['S60 speed (km/h)'].values
S60_GT=df_SVR['S60 speed (km/h) ground truth'].values
S61_SVR=df_SVR['S61 speed (km/h)'].values
S61_CLASS=df_CLASS['S61 speed (km/h)'].values
S61_GT=df_SVR['S61 speed (km/h) ground truth'].values

GT=np.concatenate((S54_GT,S1706_GT,S56_GT,S57_GT,S1707_GT,S59_GT,S60_GT, S61_GT), axis=None)
SVR=np.concatenate((S54_SVR,S1706_SVR,S56_SVR,S57_SVR,S1707_SVR,S59_SVR,S60_SVR, S61_SVR), axis=None)
CLASS=np.concatenate((S54_CLASS,S1706_CLASS,S56_CLASS,S57_CLASS,S1707_CLASS,S59_CLASS,S60_CLASS, S61_CLASS), axis=None)
len(GT)
error_SVR=math.sqrt(mean_squared_error(GT,SVR))
round(error_SVR,4)
error_CLASS=math.sqrt(mean_squared_error(GT,CLASS))
round(error_CLASS,4)


plt.figure(figsize=(35,25))
plt.subplot(2,4,1)
plt.plot(x,S54_SVR,'r-',label='SVR prediction')
plt.plot(x,S54_CLASS,'b-',label='Classification prediction')
plt.plot(x,S54_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S54')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,2)
plt.plot(x,S1706_SVR,'r-',label='SVR prediction')
plt.plot(x,S1706_CLASS,'b-',label='Classification prediction')
plt.plot(x,S1706_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1706')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,3)
plt.plot(x,S56_SVR,'r-',label='SVR prediction')
plt.plot(x,S56_CLASS,'b-',label='Classification prediction')
plt.plot(x,S56_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S56')
plt.xticks(rotation=30,size=8)
plt.legend(loc='lower right')
plt.subplot(2,4,4)
plt.plot(x,S57_SVR,'r-',label='SVR prediction')
plt.plot(x,S57_CLASS,'b-',label='Classification prediction')
plt.plot(x,S57_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S57')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,5)
plt.plot(x,S1707_SVR,'r-',label='SVR prediction')
plt.plot(x,S1707_CLASS,'b-',label='Classification prediction')
plt.plot(x,S1707_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1707')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,6)
plt.plot(x,S59_SVR,'r-',label='SVR prediction')
plt.plot(x,S59_CLASS,'b-',label='Classification prediction')
plt.plot(x,S59_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S59')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,7)
plt.plot(x,S60_SVR,'r-',label='SVR prediction')
plt.plot(x,S60_CLASS,'b-',label='Classification prediction')
plt.plot(x,S60_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S60')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,8)
plt.plot(x,S61_SVR,'r-',label='SVR prediction')
plt.plot(x,S61_CLASS,'b-',label='Classification prediction')
plt.plot(x,S61_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S61')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.suptitle("12/2/2014 speed predictions: loops", fontsize=24, y=0.93)
plt.show()

#########PLOTS


#take the day you would like to plot 
df_SVR = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_flow.xlsx', sheet_name='12-2-2014 morning')

df_WF = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /WALKFORWARD_SupportVectorRegression_prediction_flow.xlsx', sheet_name='12-2-2014 morning') 

#fix upper bound and lower bound for the flow 
df_WF.min(axis=0)
df_SVR.min(axis=0)
df_WF.max(axis=0)
df_SVR.max(axis=0)

minimum=4500
maximum=8500
#ground truth svr classifaction prediction 
S54_SVR=df_SVR['S54 flow (veh/h)'].values
S54_WF=df_WF['S54 flow (veh/h)'].values
S54_GT=df_SVR['S54 flow (veh/h) ground truth'].values
S1706_SVR=df_SVR['S1706 flow (veh/h)'].values
S1706_WF=df_WF['S1706 flow (veh/h)'].values
S1706_GT=df_SVR['S1706 flow (veh/h) ground truth'].values
S56_SVR=df_SVR['S56 flow (veh/h)'].values
S56_WF=df_WF['S56 flow (veh/h)'].values
S56_GT=df_SVR['S56 flow (veh/h) ground truth'].values
S57_SVR=df_SVR['S57 flow (veh/h)'].values
S57_WF=df_WF['S57 flow (veh/h)'].values
S57_GT=df_SVR['S57 flow (veh/h) ground truth'].values
S1707_SVR=df_SVR['S1707 flow (veh/h)'].values
S1707_WF=df_WF['S1707 flow (veh/h)'].values
S1707_GT=df_SVR['S1707 flow (veh/h) ground truth'].values
S59_SVR=df_SVR['S59 flow (veh/h)'].values
S59_WF=df_WF['S59 flow (veh/h)'].values
S59_GT=df_SVR['S59 flow (veh/h) ground truth'].values
S60_SVR=df_SVR['S60 flow (veh/h)'].values
S60_WF=df_WF['S60 flow (veh/h)'].values
S60_GT=df_SVR['S60 flow (veh/h) ground truth'].values
S61_SVR=df_SVR['S61 flow (veh/h)'].values
S61_WF=df_WF['S61 flow (veh/h)'].values
S61_GT=df_SVR['S61 flow (veh/h) ground truth'].values

GT=np.concatenate((S54_GT,S1706_GT,S56_GT,S57_GT,S1707_GT,S59_GT,S60_GT, S61_GT), axis=None)
SVR=np.concatenate((S54_SVR,S1706_SVR,S56_SVR,S57_SVR,S1707_SVR,S59_SVR,S60_SVR, S61_SVR), axis=None)
WF=np.concatenate((S54_WF,S1706_WF,S56_WF,S57_WF,S1707_WF,S59_WF,S60_WF, S61_WF), axis=None)
len(GT)
error_SVR=math.sqrt(mean_squared_error(GT,SVR))
round(error_SVR,4)
error_WF=math.sqrt(mean_squared_error(GT,WF))
round(error_WF,4)


plt.figure(figsize=(35,25))
plt.subplot(2,4,1)
plt.plot(x,S54_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S54_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S54_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S54')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,2)
plt.plot(x,S1706_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S1706_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S1706_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1706')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,3)
plt.plot(x,S56_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S56_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S56_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S56')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,4)
plt.plot(x,S57_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S57_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S57_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S57')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,5)
plt.plot(x,S1707_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S1707_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S1707_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1707')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,6)
plt.plot(x,S59_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S59_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S59_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S59')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,7)
plt.plot(x,S60_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S60_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S60_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S60')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,8)
plt.plot(x,S61_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S61_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S61_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('veh/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S61')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.suptitle("12/2/2014 flow predictions: loops",fontsize=24, y=0.93)
plt.show()



df_SVR = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /SupportVectorRegression_prediction_speed.xlsx', sheet_name='12-2-2014 morning')

df_WF = pd.read_excel('/Users/nronzoni/Desktop/TrafficData Minnesota/Prediction without ramps /WALKFORWARD_SupportVectorRegression_prediction_speed.xlsx', sheet_name='12-2-2014 morning') 

#fix upper bound and lower bound for the flow 
df_WF.min(axis=0)
df_SVR.min(axis=0)
df_WF.max(axis=0)
df_SVR.max(axis=0)

minimum=10
maximum=125
#ground truth svr classifaction prediction 
S54_SVR=df_SVR['S54 speed (km/h)'].values
S54_WF=df_WF['S54 speed (km/h)'].values
S54_GT=df_SVR['S54 speed (km/h) ground truth'].values
S1706_SVR=df_SVR['S1706 speed (km/h)'].values
S1706_WF=df_WF['S1706 speed (km/h)'].values
S1706_GT=df_SVR['S1706 speed (km/h) ground truth'].values
S56_SVR=df_SVR['S56 speed (km/h)'].values
S56_WF=df_WF['S56 speed (km/h)'].values
S56_GT=df_SVR['S56 speed (km/h) ground truth'].values
S57_SVR=df_SVR['S57 speed (km/h)'].values
S57_WF=df_WF['S57 speed (km/h)'].values
S57_GT=df_SVR['S57 speed (km/h) ground truth'].values
S1707_SVR=df_SVR['S1707 speed (km/h)'].values
S1707_WF=df_WF['S1707 speed (km/h)'].values
S1707_GT=df_SVR['S1707 speed (km/h) ground truth'].values
S59_SVR=df_SVR['S59 speed (km/h)'].values
S59_WF=df_WF['S59 speed (km/h)'].values
S59_GT=df_SVR['S59 speed (km/h) ground truth'].values
S60_SVR=df_SVR['S60 speed (km/h)'].values
S60_WF=df_WF['S60 speed (km/h)'].values
S60_GT=df_SVR['S60 speed (km/h) ground truth'].values
S61_SVR=df_SVR['S61 speed (km/h)'].values
S61_WF=df_WF['S61 speed (km/h)'].values
S61_GT=df_SVR['S61 speed (km/h) ground truth'].values

GT=np.concatenate((S54_GT,S1706_GT,S56_GT,S57_GT,S1707_GT,S59_GT,S60_GT, S61_GT), axis=None)
SVR=np.concatenate((S54_SVR,S1706_SVR,S56_SVR,S57_SVR,S1707_SVR,S59_SVR,S60_SVR, S61_SVR), axis=None)
WF=np.concatenate((S54_WF,S1706_WF,S56_WF,S57_WF,S1707_WF,S59_WF,S60_WF, S61_WF), axis=None)
len(GT)
error_SVR=math.sqrt(mean_squared_error(GT,SVR))
round(error_SVR,4)
error_WF=math.sqrt(mean_squared_error(GT,WF))
round(error_WF,4)

              
plt.figure(figsize=(35,25))
plt.subplot(2,4,1)
plt.plot(x,S54_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S54_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S54_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S54')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,2)
plt.plot(x,S1706_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S1706_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S1706_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1706')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,3)
plt.plot(x,S56_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S56_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S56_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S56')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,4)
plt.plot(x,S57_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S57_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S57_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S57')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,5)
plt.plot(x,S1707_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S1707_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S1707_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S1707')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,6)
plt.plot(x,S59_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S59_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S59_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S59')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,7)
plt.plot(x,S60_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S60_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S60_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S60')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.subplot(2,4,8)
plt.plot(x,S61_SVR,'r-',label='SVR multistep prediction')
plt.plot(x,S61_WF,'b-',label='SVR Walk Forward prediction')
plt.plot(x,S61_GT,'g-',label='ground truth')
plt.xlabel('hours of the day')
plt.ylabel('km/h',labelpad=0)
plt.ylim((minimum,maximum))
plt.title('S61')
plt.xticks(rotation=30,size=8)
plt.legend(loc='upper right')
plt.suptitle("12/2/2014 speed predictions: loops", fontsize=24, y=0.93)
plt.show()







