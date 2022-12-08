"""ARIMAXS.py: ARIMAXS: ARIMAX with semantic information."""
#Major update: for ARIMAXS
__author__ = "Wanarat J"
__email__ = "wanaratj15@gmail.com"
__version__ = "1.1.0"
__date__ = "2020-08-06"
_update_= "2022-04-12"

import math
import pandas as pd
import numpy as np
import statistics 
from datetime import datetime
from pandas import DataFrame
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from owlready2 import *

#====================================================================	
#1.read csv to dataframe
def loadTsData(province,data_path,status):
	ts = pd.read_csv(data_path, low_memory=False, header=0)
	
	#normalization
	collist = ['density', 'tourist', 'pm25','pm10','inflection','death']
	n_ts = normalization(ts,collist)
	
	#get col name
	colname = list(n_ts.columns.values.tolist()) 
	
	#select only place to consider 
	if status == 1:
		n_filter_ts = n_ts[n_ts['provinceEn'].notnull() &(n_ts['provinceEn']== province)]
	elif status == 0:
		n_filter_ts = n_ts[n_ts['provinceEn'].notnull() &(n_ts['provinceEn']== province)]
		n_filter_ts = n_filter_ts.head(1)
	
	return n_ts,n_filter_ts,colname
	
#====================================================================
#2.normalization data
def normalization(df,collist):
	#https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
	for i in collist:
		df[i] = (df[i]-df[i].min()) / (df[i].max()-df[i].min())
	return df	
	
#====================================================================	
#3. load ontology
def loadOntology(IRI):
	onto = get_ontology(IRI).load()
	return onto
	
#====================================================================	
#4.Semantic Enhancement 
#4.1 Calculate Sementic Distance from ontology
def getLCS(c1,c2,anc_c1,anc_c2,root):		
	#get lowest common subsumer
	a_dict = {}
	for i in anc_c1:
		for j in anc_c2:
			if i == j:
				if i not in a_dict:
					count1 = 0
					com1 = c1.is_a[0]
					while com1 != i and com1 != root:
						count1= count1+1
						com1 = com1.is_a[0]
					count2 = 0
					com2 = c2.is_a[0]
					while com2 != j and com2 != root :
						count2= count2+1
						com2 = com2.is_a[0]
					path = count1+count2+1
					a_dict[i]=path
	#select shortest path
	shortest_path = min(a_dict, key=a_dict.get)#get key of dict
	return a_dict[shortest_path]
	
def calSemSimByPath_List(sem_list,colname_list,ontology):
	k=10 #constance->max possible path btw two concepts
	a_dict = {}
	for sem in sem_list:
		cur = ontology.search(label=sem)
		anc_cur_list = list(cur[0].ancestors())
		for i in colname_list:
			path = 0
			dest = ontology.search(label=i)
			if dest == []:
				sim_path = 0/k
				a_dict[i]=sim_path
			elif cur[0] == dest[0]:
				sim_path = 1/k
				a_dict[i]=sim_path
			else:
				anc_dest_list = list(dest[0].ancestors())
				path = getLCS(cur[0],dest[0],anc_cur_list,anc_dest_list,ontology.entity)
				sim_path = path/k
				if i not in a_dict:#check and store the less weight_dict
					a_dict[i]=sim_path
				else:
					sim_path_old=a_dict[i]
					if sim_path_old > sim_path:
						a_dict[i]=sim_path
			
	return a_dict

#4.2 Calculate Similarity by Euclid Distance and Semantic Distance
def calEuclidDistanceWithSemWeight(x_row,y_row,weight_dict):
	x = list(x_row)
	y = list(y_row)
	z = list(weight_dict.values())
	distance = math.sqrt(sum([(a-b)**2*c for a,b,c in zip(x,y,z)if type(a) is not str]))
	return distance

#4.3 Select related items with Similarity	
def selectBySemSim(row_semantic, consider_list,ts,dict_weight,threshold):
	a_dict = {}
	count = 0
	for index,row in consider_list.iterrows():
		if threshold >= calEuclidDistanceWithSemWeight(row,row_semantic,dict_weight):
			if row_semantic["txn_date"] == row["txn_date"] and row_semantic["provinceEn"] == row["provinceEn"]:
				continue
			else:
				row['ref_date'] = row_semantic["txn_date"]
				row['ref_prov'] = row_semantic["provinceEn"]
				ts = ts.append(row, ignore_index=True)
				count=count+1
	return ts


def filterBySemSim(all_ts,ts,dict_weight,threshold):
	no = len(ts)
	remain = no
	ts["ref_date"] = ts["txn_date"]
	ts["ref_prov"] = ts["provinceEn"]
	result = ts
	all_ts["ref_date"] = np.nan
	all_ts["ref_prov"] = np.nan
	for index,row in ts.iterrows():
		result=selectBySemSim(row,all_ts,result,dict_weight,threshold) #select the group of row
		remain = remain-1
	return result

def semanticEnhancement(threshold,province,exog):
	province = province
	sem = exog #exogenous data
	
	#semantic data
	data_path = 'input_all_province.csv'
	all_ts,ts,colname = loadTsData(province,data_path)
	
	#semantic meta data
	onto_path = "file://COVID-arimaxs.owl"
	onto = loadOntology(onto_path)

	#enhancement
	threshold = threshold
	simpath = calSemSimByPath_List(sem, colname, onto)
	ts = filterBySemSim(all_ts,ts,simpath,threshold)
	return ts

	
#====================================================================	
#5.Prediction
def forecast(ts,p,d,q,testsize):

	Y = ts['inflection'].values
	X = ts[['pm25']].values

	size = int(len(Y) * testsize)
	train, test = Y,Y[size:len(Y)]
	history = [x for x in train]
	Xtrain, Xtest = X,X[size:len(X)]
	Xhistory = [x for x in Xtrain]
	predictions = list()
	lower = list()
	upper = list()
	for t in range(len(test)):
		try:
			model = ARIMA(endog=history, exog=Xhistory, order=(p,d,q))
			model_fit = model.fit(disp=0)
			output = model_fit.forecast(exog=[Xhistory[t]])
			yhat = output[0]
			if np.isnan(yhat):
				yhat=0
			if yhat[0] < 0: 
				yhat = np.array([0])
			predictions.append(yhat)
			conf = output[2][0]
			lower.append(np.array([conf[0]]))
			upper.append(np.array([conf[1]]))
			obs = float(test[t])
			Xobs = Xtest[t]
			history.append(obs)
			Xhistory.append(Xobs)
		except:	
			predictions.append(yhat)
			conf = output[2][0]
			lower.append(np.array([conf[0]]))
			upper.append(np.array([conf[1]]))
			obs = test[t]
			Xobs = Xtest[t]
			history.append(obs)
			Xhistory.append(Xobs)
		
	error_mse = mean_squared_error(test, predictions)
	error_mae = mean_absolute_error(test,predictions)
	error_rmse = mean_squared_error(test, predictions, squared=False)
	error_mape = mean_absolute_percentage_error(test,predictions)
	print('ARIMA Test MSE: %.4f' % error_mse)
	print('ARIMA Test RMSE: %.4f' % error_rmse)
	print('ARIMA Test MAE: %.4f' % error_mae)
	print('ARIMA Test MAPE:', error_mape)
	
	return test,predictions,error_mse

#====================================================================		

if __name__ == "__main__":
	
	province="Bangkok"
	threshold = 0.01
	exog =["tourist"]
	ts = semanticEnhancement(threshold,province,exog)
	
	p=3
	d=1
	q=5
	size = 0.3
	path ='input_all_province.csv'
	all_ts,ts,colname = loadTsData(province,path)
	forecast(ts,p,d,q,size)
	
